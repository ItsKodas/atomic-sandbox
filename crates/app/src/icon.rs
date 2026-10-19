//! The app's icon: the title bar's mark (`ui::menus::brand`) drawn at any size — a violet halo,
//! a violet core and a white highlight in an 18-unit box. One drawing for the window and taskbar
//! icon (`main`) and the `.ico` embedded in `app.exe` (`build.rs` includes this file), so they
//! cannot drift apart. Pure std: no egui, no winit.

/// (centre x, centre y, radius) in the 18-unit box, and the straight RGBA colour.
const CIRCLES: [((f32, f32, f32), [f32; 4]); 3] = [
    ((9.0, 9.0, 9.0), [0x8b as f32 / 255.0, 0x5c as f32 / 255.0, 0xf6 as f32 / 255.0, 0.25]), // theme::GLOW at 25 %
    ((9.0, 9.0, 6.0), [0xa7 as f32 / 255.0, 0x8b as f32 / 255.0, 0xfa as f32 / 255.0, 0.8]),  // theme::ACCENT at 80 %
    ((8.0, 8.0, 2.2), [1.0, 1.0, 1.0, 1.0]),                                                   // the highlight
];

/// The mark as straight-alpha RGBA rows, top row first, `size` × `size`, 4 × 4 supersampled.
pub fn rgba(size: u32) -> Vec<u8> {
    const SS: u32 = 4;
    let scale = 18.0 / size as f32;
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    for py in 0..size {
        for px in 0..size {
            // premultiplied accumulation over the subsamples
            let mut acc = [0.0f32; 4];
            for sy in 0..SS {
                for sx in 0..SS {
                    let x = (px as f32 + (sx as f32 + 0.5) / SS as f32) * scale;
                    let y = (py as f32 + (sy as f32 + 0.5) / SS as f32) * scale;
                    let mut c = [0.0f32; 4]; // premultiplied
                    for ((cx, cy, r), col) in CIRCLES {
                        if (x - cx).powi(2) + (y - cy).powi(2) <= r * r {
                            let a = col[3];
                            for k in 0..3 {
                                c[k] = col[k] * a + c[k] * (1.0 - a);
                            }
                            c[3] = a + c[3] * (1.0 - a);
                        }
                    }
                    for k in 0..4 {
                        acc[k] += c[k];
                    }
                }
            }
            let n = (SS * SS) as f32;
            let a = acc[3] / n;
            for k in 0..3 {
                let v = if a > 0.0 { acc[k] / n / a } else { 0.0 };
                out.push((v * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
            }
            out.push((a * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
        }
    }
    out
}

/// A Windows `.ico` holding the mark at `sizes` (32-bit BMP entries, alpha in the colour data).
#[allow(dead_code)] // build.rs only
pub fn ico(sizes: &[u32]) -> Vec<u8> {
    let mut images: Vec<Vec<u8>> = Vec::new();
    for &s in sizes {
        let px = rgba(s);
        let mut b = Vec::new();
        // BITMAPINFOHEADER; the height counts the colour rows and the AND mask's
        for v in [40u32, s, s * 2] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&1u16.to_le_bytes()); // planes
        b.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
        for _ in 0..6 {
            b.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB, sizes, resolution, palette
        }
        for y in (0..s).rev() {
            for x in 0..s {
                let i = ((y * s + x) * 4) as usize;
                b.extend_from_slice(&[px[i + 2], px[i + 1], px[i], px[i + 3]]); // BGRA
            }
        }
        let mask_row = s.div_ceil(32) * 4;
        b.extend(std::iter::repeat_n(0u8, (mask_row * s) as usize)); // all opaque: alpha decides
        images.push(b);
    }
    let mut out = Vec::new();
    out.extend_from_slice(&[0, 0, 1, 0]); // reserved, type 1 = icon
    out.extend_from_slice(&(sizes.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * sizes.len() as u32;
    for (&s, img) in sizes.iter().zip(&images) {
        let d = if s >= 256 { 0 } else { s as u8 };
        out.extend_from_slice(&[d, d, 0, 0]);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(&(img.len() as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += img.len() as u32;
    }
    for img in images {
        out.extend_from_slice(&img);
    }
    out
}
