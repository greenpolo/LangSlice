mod atlas;
mod commands;
mod ollama;
mod ollama_models;

use std::sync::Mutex;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(commands::AppState::new()))
        .manage(Mutex::new(ollama::OllamaState::default()))
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            commands::list_atlases,
            commands::load_atlas,
            commands::get_atlas_info,
            commands::get_coronal_slice,
            commands::get_border_volume,
            commands::get_brain_mesh,
            commands::get_all_volumes,
            commands::run_estimate,
            commands::run_register,
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
