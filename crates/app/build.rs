//! Embeds the app's name (the version resource: Task Manager shows "Atomic Sandbox") and icon (`src/icon.rs`, the title bar's mark) in `app.exe` on Windows, so
//! Explorer and shortcuts show it. A missing resource compiler only warns: the window still sets
//! its icon at run time.

#[path = "src/icon.rs"]
mod icon;

fn main() {
    println!("cargo:rerun-if-changed=src/icon.rs");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let path = out.join("atomic-sandbox.ico");
    std::fs::write(&path, icon::ico(&[16, 20, 24, 32, 40, 48, 64, 256])).expect("write the icon");
    let mut res = winresource::WindowsResource::new();
    res.set_icon(path.to_str().expect("utf-8 OUT_DIR"));
    // Task Manager's Processes tab shows FileDescription (else the crate's name, "app")
    res.set("FileDescription", "Atomic Sandbox");
    res.set("ProductName", "Atomic Sandbox");
    res.set("LegalCopyright", "Copyright (c) 2026 Dakoda Lancelot (ItsKodas)");
    if let Err(e) = res.compile() {
        println!("cargo:warning=the exe icon was not embedded ({e}); the window icon is unaffected");
    }
}
