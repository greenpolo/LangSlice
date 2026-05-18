mod atlas;
mod commands;
mod litert_lm;
mod local_engines;
mod ollama;
mod ollama_models;

use std::sync::Mutex;

use tauri::{image::Image, Manager, Theme, WindowEvent};

const ICON_LIGHT_THEME: &[u8] = include_bytes!("../icons/icon.ico");
const ICON_DARK_THEME: &[u8] = include_bytes!("../icons/icon-dark.ico");

fn icon_for_theme(theme: Theme) -> Result<Image<'static>, tauri::Error> {
    // Windows DARK taskbar -> lighter (sky-blue) icon; LIGHT taskbar -> saturated icon.
    let bytes = match theme {
        Theme::Dark => ICON_DARK_THEME,
        _ => ICON_LIGHT_THEME,
    };
    Image::from_bytes(bytes)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(commands::AppState::new()))
        .manage(Mutex::new(ollama::OllamaState::default()))
        .manage(Mutex::new(litert_lm::LiteRtLmState::default()))
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let initial_theme = window.theme().unwrap_or(Theme::Light);
                if let Ok(icon) = icon_for_theme(initial_theme) {
                    let _ = window.set_icon(icon);
                }
                let window_handle = window.clone();
                window.on_window_event(move |event| {
                    if let WindowEvent::ThemeChanged(theme) = event {
                        if let Ok(icon) = icon_for_theme(*theme) {
                            let _ = window_handle.set_icon(icon);
                        }
                    }
                });
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_atlases,
            commands::list_available_atlases,
            commands::download_atlas,
            local_engines::probe_local_engines,
            local_engines::probe_custom_local_endpoint,
            commands::load_atlas,
            commands::get_atlas_info,
            commands::get_coronal_slice,
            commands::get_border_volume,
            commands::get_brain_mesh,
            commands::get_all_volumes,
            commands::run_estimate,
            commands::run_register,
            commands::run_quick_affine,
            commands::run_export,
            commands::read_env_file,
            commands::write_env_file,
            commands::load_slice_image,
            commands::precache_images,
            commands::scan_image_folder,
            ollama::ollama_status,
            ollama::ollama_start,
            ollama::ollama_stop,
            ollama_models::ollama_list_models,
            ollama_models::ollama_model_info,
            ollama_models::ollama_pull_model,
            ollama_models::ollama_delete_model,
            ollama_models::ollama_running_models,
            ollama_models::ollama_available_models,
            litert_lm::litert_lm_status,
            litert_lm::litert_lm_start,
            litert_lm::litert_lm_stop,
            litert_lm::litert_lm_download_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
