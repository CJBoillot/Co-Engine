//! Build script: bake the CoE logo into the Windows executable so the taskbar,
//! Alt-Tab, and file-explorer icons all show it consistently (the runtime winit
//! window icon only reliably covers the title bar).
//!
//! Converts assets/icon.png -> a 256px .ico in OUT_DIR, then embeds it via
//! winresource. No-op on non-Windows targets.

fn main() {
    println!("cargo:rerun-if-changed=assets/icon.png");
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(target_os = "windows")]
    {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        let ico_path = std::path::Path::new(&out_dir).join("icon.ico");

        match image::open("assets/icon.png") {
            Ok(img) => {
                // ICO images max out at 256px on a side.
                let resized = img.resize(256, 256, image::imageops::FilterType::Lanczos3);
                if let Err(e) = resized.save(&ico_path) {
                    println!("cargo:warning=failed to write icon.ico: {e}");
                    return;
                }
                let mut res = winresource::WindowsResource::new();
                res.set_icon(ico_path.to_str().expect("icon path utf8"));
                if let Err(e) = res.compile() {
                    println!("cargo:warning=failed to embed exe icon: {e}");
                }
            }
            Err(e) => {
                println!("cargo:warning=assets/icon.png not found, skipping exe icon: {e}");
            }
        }
    }
}
