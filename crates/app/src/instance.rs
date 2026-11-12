//! Another running copy of this install of the app (installer Task 4 fix round 1, ruling I3): the
//! silent installer must never close another Atomic Sandbox window — whatever runs there would
//! stop without a click there — so Restart now refuses while one is open.
//!
//! Each window holds a named mutex for its lifetime ([`claim`], at start-up): `Local\AtomicSandbox-
//! <hash of the exe's folder>`, so a build from source, a portable copy and the installed build
//! never see each other (they don't replace each other's files either). Before the hand-off
//! [`another_copy_running`] closes this process's own handle and opens the name again: the open
//! succeeds only while some other process still holds a handle to it. This process's handle is then
//! taken again. Windows only; elsewhere there is no other copy to find.
use std::path::Path;

/// The refusal (`Viewer::restart_for_update`).
pub const OTHER_COPY: &str = "Close the other Atomic Sandbox window first, then press Restart now again.";

/// The mutex name for an exe in `exe_dir`: `Local\AtomicSandbox-<16 hex digits>`, the FNV-1a hash of
/// the folder's path, lower-cased (Windows paths ignore case). A kernel object name may not hold
/// another `\`, so the path itself is never in it.
pub fn name_for(exe_dir: &Path) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in exe_dir.to_string_lossy().to_lowercase().bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("Local\\AtomicSandbox-{h:016x}")
}

/// This process's exe folder's name ([`name_for`]); `None` when the exe's path can't be read.
fn own_name() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    Some(name_for(exe.parent()?))
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ACCESS_DENIED, HANDLE};
    use windows_sys::Win32::System::Threading::{CreateMutexW, OpenMutexW, SYNCHRONIZATION_SYNCHRONIZE};

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// A handle to the named mutex, held until dropped.
    pub struct Held {
        name: String,
        handle: HANDLE,
    }

    // the handle is a kernel handle, usable from any thread
    unsafe impl Send for Held {}

    impl Held {
        pub fn create(name: &str) -> Option<Held> {
            let w = wide(name);
            // not owned: only its existence counts
            let handle = unsafe { CreateMutexW(std::ptr::null(), 0, w.as_ptr()) };
            (!handle.is_null()).then(|| Held { name: name.to_string(), handle })
        }

        /// Whether another handle to the name is open (another process, or another [`Held`]):
        /// this one is closed, the name opened (it exists only while some handle is open), and
        /// this one taken again.
        pub fn others(&mut self) -> bool {
            unsafe { CloseHandle(self.handle) };
            let other = others_hold(&self.name);
            let w = wide(&self.name);
            self.handle = unsafe { CreateMutexW(std::ptr::null(), 0, w.as_ptr()) };
            other
        }
    }

    impl Drop for Held {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { CloseHandle(self.handle) };
            }
        }
    }

    /// Whether any handle to `name` is open (a refused open counts: the object exists).
    pub fn others_hold(name: &str) -> bool {
        let w = wide(name);
        let h = unsafe { OpenMutexW(SYNCHRONIZATION_SYNCHRONIZE, 0, w.as_ptr()) };
        if h.is_null() {
            return unsafe { GetLastError() } == ERROR_ACCESS_DENIED;
        }
        unsafe { CloseHandle(h) };
        true
    }
}

#[cfg(not(windows))]
mod imp {
    pub struct Held;
    impl Held {
        pub fn create(_: &str) -> Option<Held> {
            Some(Held)
        }
        pub fn others(&mut self) -> bool {
            false
        }
    }
    pub fn others_hold(_: &str) -> bool {
        false
    }
}

pub use imp::Held;

static OWN: std::sync::Mutex<Option<Held>> = std::sync::Mutex::new(None);

/// Takes this window's handle (`main`, the window only), for the process's lifetime.
pub fn claim() {
    if let Some(h) = own_name().and_then(|n| Held::create(&n)) {
        *OWN.lock().unwrap_or_else(|e| e.into_inner()) = Some(h);
    }
}

/// Another copy of this install is running (module doc). A process that never claimed (headless,
/// the tests) only looks.
pub fn another_copy_running() -> bool {
    let mut own = OWN.lock().unwrap_or_else(|e| e.into_inner());
    match own.as_mut() {
        Some(h) => h.others(),
        None => own_name().is_some_and(|n| imp::others_hold(&n)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The name is per exe folder (case-insensitive) and a valid object name; a lone holder sees no
    /// other, a second holder is seen, and once it closes it is gone again.
    #[test]
    fn a_second_holder_of_the_name_is_seen() {
        let a = name_for(Path::new(r"C:\Users\Zoë Q\AppData\Local\Programs\Atomic Sandbox"));
        assert_eq!(a, name_for(Path::new(r"c:\users\zoë q\appdata\local\programs\atomic sandbox")));
        assert_ne!(a, name_for(Path::new(r"D:\Projects\Physics\target\release")));
        assert!(a.starts_with("Local\\AtomicSandbox-") && a.matches('\\').count() == 1 && a.len() == "Local\\AtomicSandbox-".len() + 16, "{a}");
        // a name of this test's own: no other process uses it
        let name = format!("Local\\AtomicSandbox-test-{}", std::process::id());
        assert!(!imp::others_hold(&name), "nobody holds it yet");
        let mut mine = Held::create(&name).expect("created");
        assert!(!mine.others(), "only this holder");
        let other = Held::create(&name).expect("a second holder (as another window would be)");
        #[cfg(windows)]
        assert!(mine.others(), "the other holder is seen");
        drop(other);
        assert!(!mine.others(), "the other closed");
        assert!(imp::others_hold(&name), "this holder is taken again after the check");
        drop(mine);
        assert!(!imp::others_hold(&name));
    }
}
