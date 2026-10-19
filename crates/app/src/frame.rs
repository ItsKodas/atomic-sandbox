//! The custom window frame (user request 2026-10-18, option B): the window has no native caption;
//! the top bar is the title bar — the menus, the scene chip and the search sit in it, with the
//! app's own minimise / maximise / close buttons at its right end. Native behaviour is kept by
//! answering Windows' hit test (`WM_NCHITTEST`) from the bar's layout: the bar's empty space is
//! `HTCAPTION` (drag, double-click maximises, Aero Snap by drag, the system menu on right-click),
//! the window's edges are the resize borders, and the maximise button is `HTMAXBUTTON`, which is
//! what makes Windows 11 show the Snap Layouts flyout on hover. While the pointer is on
//! `HTMAXBUTTON` Windows owns it (egui sees the cursor leave), so its hover and press are tracked
//! here from the non-client messages and drawn by the bar from [`max_state`].
//!
//! winit draws nothing (`with_decorations(false)`); it keeps the shadow
//! (`with_undecorated_shadow`), the rounded corners (`with_corner_preference`) and fits a
//! maximised window to the monitor's work area. The layout is shared through one process-wide
//! state: the UI writes the bar's rects each frame ([`publish`]), the subclass procedure reads
//! them. Headless runs (screenshots, replays, the UI tests) never install it, so [`active`] is
//! false there and the bar draws as before. Display only: nothing here reaches the worker or is
//! recorded. `ATOMIC_SANDBOX_NATIVE_FRAME=1` keeps the native caption.

use egui::Rect;
use std::sync::Mutex;

/// What the bar's own buttons ask of the window (performed by the window after the frame).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmd {
    Minimise,
    ToggleMaximise,
    Close,
}

struct State {
    active: bool,
    /// The bar, in points.
    bar: Rect,
    /// The bar's interactive widgets (menus, chip, search, the minimise/close buttons), in points.
    hot: Vec<Rect>,
    /// The maximise button, in points (answered `HTMAXBUTTON`).
    max_button: Rect,
    /// Pixels per point when the layout was published.
    ppp: f32,
    maximised: bool,
    max_hover: bool,
    max_down: bool,
    cmd: Option<Cmd>,
}

static STATE: Mutex<State> = Mutex::new(State {
    active: false,
    bar: Rect::NOTHING,
    hot: Vec::new(),
    max_button: Rect::NOTHING,
    ppp: 1.0,
    maximised: false,
    max_hover: false,
    max_down: false,
    cmd: None,
});

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The window has the custom frame (false headless and with the native caption).
pub fn active() -> bool {
    state().active
}

/// The bar's layout for this frame, in points.
pub fn publish(bar: Rect, hot: Vec<Rect>, max_button: Rect, ppp: f32) {
    let mut s = state();
    s.bar = bar;
    s.hot = hot;
    s.max_button = max_button;
    s.ppp = ppp;
}

/// (maximised, hover, pressed) of the maximise button.
pub fn max_state() -> (bool, bool, bool) {
    let s = state();
    (s.maximised, s.max_hover, s.max_down)
}

pub fn set_maximised(m: bool) {
    state().maximised = m;
}

pub fn request(c: Cmd) {
    state().cmd = Some(c);
}

pub fn take_cmd() -> Option<Cmd> {
    state().cmd.take()
}

/// Whether the native caption was asked for.
pub fn native_requested() -> bool {
    std::env::var("ATOMIC_SANDBOX_NATIVE_FRAME").is_ok_and(|v| v == "1")
}

#[cfg(windows)]
pub use win::{exe_icon, install};

/// No custom frame off Windows: the native decorations stay.
#[cfg(not(windows))]
pub fn install(_window: &winit::window::Window) -> bool {
    false
}

#[cfg(windows)]
mod win {
    use super::state;
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
    use windows_sys::Win32::Graphics::Gdi::{InvalidateRect, ScreenToClient};
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{TrackMouseEvent, TME_LEAVE, TME_NONCLIENT, TRACKMOUSEEVENT};
    use windows_sys::Win32::UI::Shell::{DefSubclassProc, SetWindowSubclass};
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetClientRect, IsZoomed, ShowWindow, SW_MAXIMIZE, SW_RESTORE};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    const WM_NCHITTEST: u32 = 0x0084;
    const WM_MOUSEMOVE: u32 = 0x0200;
    const WM_NCMOUSEMOVE: u32 = 0x00A0;
    const WM_NCLBUTTONDOWN: u32 = 0x00A1;
    const WM_NCLBUTTONUP: u32 = 0x00A2;
    const WM_NCLBUTTONDBLCLK: u32 = 0x00A3;
    const WM_NCMOUSELEAVE: u32 = 0x02A2;
    const HTCLIENT: usize = 1;
    const HTCAPTION: usize = 2;
    const HTMAXBUTTON: usize = 9;
    const HTLEFT: usize = 10;
    const HTRIGHT: usize = 11;
    const HTTOP: usize = 12;
    const HTTOPLEFT: usize = 13;
    const HTTOPRIGHT: usize = 14;
    const HTBOTTOM: usize = 15;
    const HTBOTTOMLEFT: usize = 16;
    const HTBOTTOMRIGHT: usize = 17;
    /// The resize border inside the window's edge, in points.
    const BORDER_PT: f32 = 6.0;

    /// The exe's own icon (resource 1, embedded by `build.rs`) as the window's big and small icon
    /// and its class icon, loaded by Windows at the sizes it asks for: the taskbar shows the
    /// generic icon with only winit's RGBA icons. False when the exe carries no icon.
    pub fn exe_icon(window: &winit::window::Window) -> bool {
        use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            GetSystemMetrics, LoadImageW, SendMessageW, SetClassLongPtrW, GCLP_HICON, GCLP_HICONSM, ICON_BIG, ICON_SMALL,
            IMAGE_ICON, SM_CXICON, SM_CXSMICON, SM_CYICON, SM_CYSMICON, WM_SETICON,
        };
        let Ok(h) = window.window_handle() else { return false };
        let RawWindowHandle::Win32(w) = h.as_raw() else { return false };
        let hwnd = w.hwnd.get() as HWND;
        unsafe {
            let module = GetModuleHandleW(std::ptr::null());
            let load = |cx, cy| LoadImageW(module, 1 as *const u16, IMAGE_ICON, GetSystemMetrics(cx), GetSystemMetrics(cy), 0);
            let big = load(SM_CXICON, SM_CYICON);
            let small = load(SM_CXSMICON, SM_CYSMICON);
            if big.is_null() || small.is_null() {
                return false;
            }
            SendMessageW(hwnd, WM_SETICON, ICON_BIG as WPARAM, big as LPARAM);
            SendMessageW(hwnd, WM_SETICON, ICON_SMALL as WPARAM, small as LPARAM);
            SetClassLongPtrW(hwnd, GCLP_HICON, big as isize);
            SetClassLongPtrW(hwnd, GCLP_HICONSM, small as isize);
        }
        true
    }

    /// Subclasses the window's procedure; true if the custom frame is in place.
    pub fn install(window: &winit::window::Window) -> bool {
        let Ok(h) = window.window_handle() else { return false };
        let RawWindowHandle::Win32(w) = h.as_raw() else { return false };
        let hwnd = w.hwnd.get() as HWND;
        let ok = unsafe { SetWindowSubclass(hwnd, Some(proc_), 0xA70, 0) } != 0;
        if ok {
            let mut s = state();
            s.active = true;
            s.maximised = window.is_maximized();
        }
        ok
    }

    fn redraw(hwnd: HWND) {
        unsafe { InvalidateRect(hwnd, std::ptr::null(), 0) };
    }

    fn set_hover(hwnd: HWND, hover: bool) {
        let changed = {
            let mut s = state();
            let changed = s.max_hover != hover || (!hover && s.max_down);
            s.max_hover = hover;
            if !hover {
                s.max_down = false;
            }
            changed
        };
        if changed {
            redraw(hwnd);
        }
    }

    /// The hit test, `p` in client pixels.
    fn hit(hwnd: HWND, p: POINT) -> usize {
        let mut rc = RECT { left: 0, top: 0, right: 0, bottom: 0 };
        unsafe { GetClientRect(hwnd, &mut rc) };
        let zoomed = unsafe { IsZoomed(hwnd) } != 0;
        let s = state();
        let ppp = s.ppp.max(0.1);
        if !zoomed {
            let b = (BORDER_PT * ppp).round() as i32;
            let (l, r) = (p.x < rc.left + b, p.x >= rc.right - b);
            let (t, bo) = (p.y < rc.top + b, p.y >= rc.bottom - b);
            match (t, bo, l, r) {
                (true, _, true, _) => return HTTOPLEFT,
                (true, _, _, true) => return HTTOPRIGHT,
                (_, true, true, _) => return HTBOTTOMLEFT,
                (_, true, _, true) => return HTBOTTOMRIGHT,
                (true, ..) => return HTTOP,
                (_, true, ..) => return HTBOTTOM,
                (_, _, true, _) => return HTLEFT,
                (_, _, _, true) => return HTRIGHT,
                _ => {}
            }
        }
        let pt = egui::pos2(p.x as f32 / ppp, p.y as f32 / ppp);
        if s.max_button.contains(pt) {
            return HTMAXBUTTON;
        }
        if s.bar.contains(pt) && !s.hot.iter().any(|r| r.contains(pt)) {
            return HTCAPTION;
        }
        HTCLIENT
    }

    unsafe extern "system" fn proc_(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM, _id: usize, _data: usize) -> LRESULT {
        match msg {
            WM_NCHITTEST => {
                let mut p = POINT { x: (lp & 0xffff) as i16 as i32, y: ((lp >> 16) & 0xffff) as i16 as i32 };
                unsafe { ScreenToClient(hwnd, &mut p) };
                return hit(hwnd, p) as LRESULT;
            }
            WM_NCMOUSEMOVE => {
                let on = wp == HTMAXBUTTON;
                if on {
                    let mut t = TRACKMOUSEEVENT {
                        cbSize: std::mem::size_of::<TRACKMOUSEEVENT>() as u32,
                        dwFlags: TME_LEAVE | TME_NONCLIENT,
                        hwndTrack: hwnd,
                        dwHoverTime: 0,
                    };
                    unsafe { TrackMouseEvent(&mut t) };
                }
                set_hover(hwnd, on);
            }
            WM_NCMOUSELEAVE | WM_MOUSEMOVE => set_hover(hwnd, false),
            WM_NCLBUTTONDOWN | WM_NCLBUTTONDBLCLK if wp == HTMAXBUTTON => {
                state().max_down = true;
                redraw(hwnd);
                return 0; // never the classic button's own press
            }
            WM_NCLBUTTONUP if wp == HTMAXBUTTON => {
                let down = std::mem::take(&mut state().max_down);
                if down {
                    let zoomed = unsafe { IsZoomed(hwnd) } != 0;
                    unsafe { ShowWindow(hwnd, if zoomed { SW_RESTORE } else { SW_MAXIMIZE }) };
                }
                redraw(hwnd);
                return 0;
            }
            _ => {}
        }
        unsafe { DefSubclassProc(hwnd, msg, wp, lp) }
    }
}
