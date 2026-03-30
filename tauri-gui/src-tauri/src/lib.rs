mod atlas;
mod commands;

use std::sync::Mutex;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(commands::AppState { atlas: None }))
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
            commands::load_slice_image,
            commands::scan_image_folder,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
