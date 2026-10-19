//! Disk and encoding for the capture, off the render thread: a few PNG writer threads behind a
//! bounded queue ([`Writer`]; a full queue refuses the frame instead of blocking), and the
//! recording's finish on its own thread ([`finish`]: fill the skipped slots with the previous frame,
//! encode with `ffmpeg` when it is on PATH — H.264, yuv420p, 30 fps — and delete the PNGs after a
//! successful encode; otherwise keep them and say how to encode them).
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A frame read back from the GPU: tightly packed RGBA8, top row first.
pub struct Image {
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

/// What the capture threads tell the viewer (shown by the buttons and in the status line).
#[derive(Clone, Debug, PartialEq)]
pub enum Note {
    /// A screenshot was written.
    Saved(PathBuf),
    /// A recording is being finished (filling, encoding).
    Finishing(String),
    /// A recording is done: what was kept, in words.
    Finished(String),
    Error(String),
    /// Whether `ffmpeg` is on PATH (the probe's answer).
    Ffmpeg(bool),
}

/// Writes one PNG (fast compression: a 4K frame in a few tens of ms).
pub fn write_png(path: &Path, img: &Image) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), img.w, img.h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut w = enc.write_header().map_err(|e| format!("{}: {e}", path.display()))?;
    w.write_image_data(&img.rgba).map_err(|e| format!("{}: {e}", path.display()))
}

/// The frame file of index `i` in a recording's folder (ffmpeg's `frame_%06d.png`).
pub fn frame_path(dir: &Path, i: u64) -> PathBuf {
    dir.join(format!("frame_{i:06}.png"))
}

/// The recording's PNG writers.
pub struct Writer {
    tx: Option<SyncSender<(u64, Image)>>,
    threads: Vec<JoinHandle<()>>,
    errors: Arc<Mutex<Vec<String>>>,
}

/// Frames that may wait for a writer (a 4K frame is 33 MB: at most ≈ 0.3 GB queued).
pub const QUEUE: usize = 8;

impl Writer {
    pub fn new(dir: PathBuf) -> Writer {
        let (tx, rx) = mpsc::sync_channel::<(u64, Image)>(QUEUE);
        let rx = Arc::new(Mutex::new(rx));
        let errors = Arc::new(Mutex::new(Vec::new()));
        let n = std::thread::available_parallelism().map_or(2, |n| (n.get() / 4).clamp(1, 4));
        let threads = (0..n)
            .map(|k| {
                let (rx, errors, dir) = (rx.clone(), errors.clone(), dir.clone());
                std::thread::Builder::new()
                    .name(format!("capture-writer-{k}"))
                    .spawn(move || loop {
                        let job = rx.lock().map(|r| r.recv());
                        let Ok(Ok((i, img))) = job else { return };
                        if let Err(e) = write_png(&frame_path(&dir, i), &img) {
                            errors.lock().map(|mut v| v.push(e)).ok();
                        }
                    })
                    .expect("spawn a capture writer")
            })
            .collect();
        Writer { tx: Some(tx), threads, errors }
    }

    /// Queues frame `i`; false when the writers are behind (the frame is dropped, its slot filled
    /// with the previous frame at the finish).
    pub fn try_send(&self, i: u64, img: Image) -> bool {
        match self.tx.as_ref().map(|tx| tx.try_send((i, img))) {
            Some(Ok(())) => true,
            Some(Err(TrySendError::Full(_) | TrySendError::Disconnected(_))) | None => false,
        }
    }

    /// Waits for the queued frames to be written; the write errors.
    fn join(mut self) -> Vec<String> {
        self.tx = None;
        for t in self.threads.drain(..) {
            t.join().ok();
        }
        self.errors.lock().map(|v| v.clone()).unwrap_or_default()
    }
}

/// What a finished recording is: its frame folder, the video path and how many frames it has.
pub struct Finish {
    pub writer: Writer,
    pub dir: PathBuf,
    pub video: PathBuf,
    pub frames: u64,
    /// A line on what was recorded (mode, sim time, notes), for the final note.
    pub summary: String,
}

/// Finishes a recording on its own thread; its notes arrive on `notes`.
pub fn finish(f: Finish, notes: Sender<Note>) {
    std::thread::Builder::new()
        .name("capture-finish".into())
        .spawn(move || {
            let note = |n: Note| {
                notes.send(n).ok();
            };
            note(Note::Finishing("writing the last frames…".into()));
            let errors = f.writer.join();
            if let Some(e) = errors.first() {
                note(Note::Error(format!("{} frame(s) could not be written: {e}", errors.len())));
            }
            let filled = match fill_gaps(&f.dir, f.frames) {
                Ok(n) => n,
                Err(e) => {
                    note(Note::Error(e));
                    return;
                }
            };
            if filled == f.frames {
                note(Note::Finished("no frames were recorded".into()));
                std::fs::remove_dir(&f.dir).ok();
                return;
            }
            let pngs = format!("{} frames in {}", f.frames, f.dir.display());
            if !ffmpeg_available() {
                note(Note::Finished(format!(
                    "{} — kept as PNGs ({pngs}; ffmpeg is not on PATH). To encode: {}",
                    f.summary,
                    encode_command(&f.dir, &f.video)
                )));
                return;
            }
            note(Note::Finishing(format!("encoding {} frames with ffmpeg…", f.frames)));
            match encode(&f.dir, &f.video) {
                Ok(()) => {
                    std::fs::remove_dir_all(&f.dir).ok();
                    note(Note::Finished(format!("{} — saved {}", f.summary, f.video.display())));
                }
                Err(e) => note(Note::Finished(format!(
                    "{} — ffmpeg failed ({e}); kept as PNGs ({pngs}). To encode: {}",
                    f.summary,
                    encode_command(&f.dir, &f.video)
                ))),
            }
        })
        .expect("spawn the capture finisher");
}

/// Copies the previous frame into every missing index below `frames` (the next one for missing
/// leading frames); returns how many were missing.
pub fn fill_gaps(dir: &Path, frames: u64) -> Result<u64, String> {
    let have: Vec<bool> = (0..frames).map(|i| frame_path(dir, i).is_file()).collect();
    let missing = have.iter().filter(|h| !**h).count() as u64;
    let Some(first) = have.iter().position(|h| *h) else { return Ok(missing) };
    let mut last = first as u64;
    for i in 0..frames {
        if have[i as usize] {
            last = i;
            continue;
        }
        let from = if i < first as u64 { first as u64 } else { last };
        std::fs::copy(frame_path(dir, from), frame_path(dir, i)).map_err(|e| format!("filling frame {i}: {e}"))?;
    }
    Ok(missing)
}

/// The ffmpeg arguments: 30 fps PNGs → H.264 in yuv420p (even dimensions: padded by a pixel when odd).
fn encode_args(dir: &Path, video: &Path) -> Vec<String> {
    let input = dir.join("frame_%06d.png");
    [
        "-y", "-hide_banner", "-loglevel", "error", "-framerate", "30", "-i",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain([input.display().to_string()])
    .chain(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"].iter().map(|s| s.to_string()))
    .chain([video.display().to_string()])
    .collect()
}

/// The command a user can run by hand (quoted paths).
pub fn encode_command(dir: &Path, video: &Path) -> String {
    encode_args(dir, video)
        .iter()
        .map(|a| if a.contains([' ', '\\', '%', '(', '+']) { format!("\"{a}\"") } else { a.clone() })
        .fold("ffmpeg".to_string(), |s, a| s + " " + &a)
}

fn command(program: &str) -> std::process::Command {
    #[allow(unused_mut)]
    let mut c = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: no console flashes up over the app
    }
    c.stdin(std::process::Stdio::null());
    c
}

pub fn ffmpeg_available() -> bool {
    command("ffmpeg").arg("-version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().is_ok_and(|s| s.success())
}

fn encode(dir: &Path, video: &Path) -> Result<(), String> {
    let out = command("ffmpeg").args(encode_args(dir, video)).output().map_err(|e| e.to_string())?;
    if out.status.success() && video.is_file() {
        Ok(())
    } else {
        let err = String::from_utf8_lossy(&out.stderr);
        Err(err.lines().last().unwrap_or("no output").trim().to_string())
    }
}

/// The notes channel's two ends.
pub fn notes() -> (Sender<Note>, Receiver<Note>) {
    mpsc::channel()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("atomic-sandbox-capture-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn gaps_are_filled_with_the_previous_frame() {
        let d = tmp("gaps");
        for (i, b) in [(1u64, b"one"), (2, b"two"), (5, b"fiv")] {
            std::fs::write(frame_path(&d, i), b).unwrap();
        }
        assert_eq!(fill_gaps(&d, 7).unwrap(), 4);
        let read = |i| std::fs::read(frame_path(&d, i)).unwrap();
        assert_eq!(read(0), b"one", "a missing first frame takes the first one written");
        assert_eq!((read(3), read(4), read(6)), (b"two".to_vec(), b"two".to_vec(), b"fiv".to_vec()));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_writers_write_and_a_full_queue_refuses() {
        let d = tmp("writer");
        let w = Writer::new(d.clone());
        let img = || Image { w: 2, h: 1, rgba: vec![255, 0, 0, 255, 0, 255, 0, 255] };
        let mut sent = 0;
        for i in 0..3 {
            sent += u64::from(w.try_send(i, img()));
        }
        assert!(sent >= 1);
        assert!(w.join().is_empty());
        let n = (0..3).filter(|&i| frame_path(&d, i).is_file()).count() as u64;
        assert_eq!(n, sent);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_hand_command_names_the_frames_and_the_video() {
        let c = encode_command(Path::new("captures/x_frames"), Path::new("captures/x.mp4"));
        assert!(c.starts_with("ffmpeg -y") && c.contains("-framerate 30") && c.contains("frame_%06d.png") && c.contains("yuv420p") && c.ends_with("captures/x.mp4"), "{c}");
    }
}
