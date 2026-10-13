//! The egui UI: tab bar, browse view (folder tree + thumbnail grid) and the
//! edit view (controls, filmstrip, image panel). Panels draw from `App`'s
//! state; widget output the panels cannot apply inline (file-dialog picks,
//! one-shot actions) is returned as `UiRequests` through egui temp-data slots.

use super::*;

/// One-shot UI output, fished out of egui's temp-data slots after the frame.
pub(crate) struct UiRequests {
    pub open_path: Option<PathBuf>,
    pub export_path: Option<PathBuf>,
    pub scan_dir: Option<PathBuf>,
    pub auto: bool,
    pub develop_raw: bool,
    pub capture_look: bool,
    pub apply_look: bool,
    pub teach_look_model: bool,
    pub trash_rejects: bool,
    pub move_rejects_dir: Option<PathBuf>,
    pub copy_picks_dir: Option<PathBuf>,
    pub classify_folder: bool,
    pub rate_folder: bool,
    pub adjust_folder: bool,
    pub look_folder: bool,
    pub similar_cull: bool,
    pub undo: bool,
    pub redo: bool,
    pub show_export: bool,
    /// A dab to place: `[x, y, radius]`, all normalised to the image.
    pub retouch: Option<[f32; 3]>,
    /// Index of the dab to remove.
    pub retouch_remove: Option<usize>,
    pub retouch_clear: bool,
}

fn take<T: 'static + Default>(ctx: &egui::Context, key: &str) -> Option<T> {
    ctx.data_mut(|d| d.remove_temp(egui::Id::new(key)))
}

pub(crate) fn take_requests(ctx: &egui::Context) -> UiRequests {
    UiRequests {
        open_path: take(ctx, KEY_OPEN_PATH),
        export_path: take(ctx, KEY_EXPORT_PATH),
        scan_dir: take(ctx, KEY_SCAN_DIR),
        auto: take::<bool>(ctx, KEY_AUTO).is_some(),
        develop_raw: take::<bool>(ctx, KEY_DEVELOP_RAW).is_some(),
        capture_look: take::<bool>(ctx, KEY_CAPTURE_LOOK).is_some(),
        apply_look: take::<bool>(ctx, KEY_APPLY_LOOK).is_some(),
        teach_look_model: take::<bool>(ctx, KEY_TEACH_LOOK_MODEL).is_some(),
        trash_rejects: take::<bool>(ctx, KEY_TRASH_REJECTS).is_some(),
        move_rejects_dir: take(ctx, KEY_MOVE_REJECTS),
        copy_picks_dir: take(ctx, KEY_COPY_PICKS),
        classify_folder: take::<bool>(ctx, KEY_CLASSIFY_FOLDER).is_some(),
        rate_folder: take::<bool>(ctx, KEY_RATE_FOLDER).is_some(),
        adjust_folder: take::<bool>(ctx, KEY_ADJUST_FOLDER).is_some(),
        look_folder: take::<bool>(ctx, KEY_LOOK_FOLDER).is_some(),
        similar_cull: take::<bool>(ctx, KEY_SIMILAR_CULL).is_some(),
        undo: take::<bool>(ctx, KEY_UNDO).is_some(),
        show_export: take::<bool>(ctx, KEY_SHOW_EXPORT).is_some(),
        redo: take::<bool>(ctx, KEY_REDO).is_some(),
        retouch: take(ctx, KEY_RETOUCH),
        retouch_remove: take(ctx, KEY_RETOUCH_REMOVE),
        retouch_clear: take::<bool>(ctx, KEY_RETOUCH_CLEAR).is_some(),
    }
}

pub(crate) fn tabs(ctx: &egui::Context, view: &mut View) {
    egui::TopBottomPanel::top("tabs").exact_height(34.0).show(ctx, |ui| {
        ui.horizontal_centered(|ui| {
            ui.add_space(4.0);
            ui.selectable_value(view, View::Browse, "  Browse  ");
            ui.selectable_value(view, View::Edit, "  Edit  ");
        });
    });
}

fn label_color(label: cull::Label) -> Option<egui::Color32> {
    match label {
        cull::Label::None => None,
        cull::Label::Red => Some(egui::Color32::from_rgb(224, 92, 92)),
        cull::Label::Yellow => Some(egui::Color32::from_rgb(225, 204, 72)),
        cull::Label::Green => Some(egui::Color32::from_rgb(94, 186, 113)),
        cull::Label::Blue => Some(egui::Color32::from_rgb(96, 152, 233)),
    }
}

fn paint_badges(p: &egui::Painter, rect: egui::Rect, meta: cull::CullMeta) {
    if meta.flag == cull::Flag::Reject {
        p.rect_filled(
            egui::Rect::from_min_size(rect.min + egui::vec2(4.0, 4.0), egui::vec2(14.0, 14.0)),
            3.0,
            egui::Color32::from_rgb(170, 60, 60),
        );
        let c = rect.min + egui::vec2(11.0, 11.0);
        let s = egui::Stroke::new(1.5_f32, egui::Color32::WHITE);
        p.line_segment([c + egui::vec2(-3.5, -3.5), c + egui::vec2(3.5, 3.5)], s);
        p.line_segment([c + egui::vec2(3.5, -3.5), c + egui::vec2(-3.5, 3.5)], s);
    } else if meta.flag == cull::Flag::Pick {
        p.rect_filled(
            egui::Rect::from_min_size(rect.min + egui::vec2(4.0, 4.0), egui::vec2(14.0, 14.0)),
            3.0,
            egui::Color32::from_rgb(66, 128, 235),
        );
        let c = rect.min + egui::vec2(11.0, 11.0);
        let s = egui::Stroke::new(1.5_f32, egui::Color32::WHITE);
        p.line_segment([c + egui::vec2(-3.5, 0.5), c + egui::vec2(-1.0, 3.5)], s);
        p.line_segment([c + egui::vec2(-1.0, 3.5), c + egui::vec2(4.0, -3.5)], s);
    }
    if meta.rating > 0 {
        p.text(
            rect.left_bottom() + egui::vec2(6.0, -6.0),
            egui::Align2::LEFT_BOTTOM,
            "★".repeat(meta.rating as usize),
            egui::FontId::proportional(10.0),
            egui::Color32::from_rgb(235, 196, 55),
        );
    }
    if let Some(color) = label_color(meta.label) {
        p.circle_filled(rect.right_top() - egui::vec2(11.0, -11.0), 4.0, color);
    }
}

// The thumbnail image plus its cull badges, shared by the browse grid and the
// edit filmstrip. `reject_overlay` tints the cell when flagged for culling.
fn paint_thumb_cell(
    p: &egui::Painter,
    img_area: egui::Rect,
    tex: Option<&egui::TextureHandle>,
    meta: cull::CullMeta,
    placeholder_size: f32,
    reject_overlay: bool,
) {
    if let Some(tex) = tex {
        let ts = tex.size_vec2();
        let s = (img_area.width() / ts.x).min(img_area.height() / ts.y).min(1.0);
        let ir = egui::Rect::from_center_size(img_area.center(), ts * s);
        p.image(
            tex.id(),
            ir,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    } else {
        p.text(
            img_area.center(),
            egui::Align2::CENTER_CENTER,
            "…",
            egui::FontId::proportional(placeholder_size),
            egui::Color32::from_gray(90),
        );
    }
    if reject_overlay && meta.flag == cull::Flag::Reject {
        p.rect_filled(
            img_area,
            0.0,
            egui::Color32::from_rgba_premultiplied(140, 20, 20, 48),
        );
    }
    paint_badges(p, img_area, meta);
}

fn show_dir_item(
    ui: &mut egui::Ui,
    path: &Path,
    depth: u8,
    expanded: &mut std::collections::HashSet<PathBuf>,
    current: Option<&Path>,
    ctx: &egui::Context,
) {
    let is_exp = expanded.contains(path);
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("/");
    let selected = current == Some(path);
    ui.horizontal(|ui| {
        ui.add_space(f32::from(depth) * 10.0);
        let (tri_rect, tri_resp) =
            ui.allocate_exact_size(egui::vec2(12.0, 14.0), egui::Sense::click());
        if ui.is_rect_visible(tri_rect) {
            let c = tri_rect.center();
            let color = egui::Color32::from_gray(110);
            let pts = if is_exp {
                vec![
                    c + egui::vec2(-4.0, -2.0),
                    c + egui::vec2(4.0, -2.0),
                    c + egui::vec2(0.0, 3.0),
                ]
            } else {
                vec![
                    c + egui::vec2(-2.0, -4.0),
                    c + egui::vec2(3.0, 0.0),
                    c + egui::vec2(-2.0, 4.0),
                ]
            };
            ui.painter()
                .add(egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE));
        }
        if tri_resp.clicked() {
            if is_exp {
                expanded.remove(path);
            } else {
                expanded.insert(path.to_owned());
            }
        }
        let color = if selected {
            egui::Color32::from_rgb(90, 140, 255)
        } else {
            egui::Color32::from_gray(195)
        };
        if ui
            .add(
                egui::Label::new(egui::RichText::new(name).small().color(color))
                    .sense(egui::Sense::click()),
            )
            .clicked()
        {
            expanded.insert(path.to_owned());
            ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_SCAN_DIR), path.to_owned()));
        }
    });
    if is_exp {
        let Ok(rd) = std::fs::read_dir(path) else {
            return;
        };
        let mut dirs: Vec<PathBuf> = rd
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_ok_and(|ft| ft.is_dir()))
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|s| !s.starts_with('.'))
            })
            .collect();
        dirs.sort_unstable();
        for dir in dirs {
            show_dir_item(ui, &dir, depth.saturating_add(1), expanded, current, ctx);
        }
    }
}

// ---- Browse view -----------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn browse(
    ctx: &egui::Context,
    app: &mut App,
    visible: &[usize],
    n_picks: usize,
    n_rejects: usize,
) {
    let grid_cols = &mut app.grid_cols;
    let tree_expanded = &mut app.tree_expanded;
    let filter = &mut app.filter;
    let min_rating = &mut app.min_rating;
    let sort = &mut app.sort;
    let grid_scroll = &mut app.flags.grid_scroll;
    let selected = &mut app.selected;
    let browse_dir = &app.browse_dir;
    let current_path = &app.current_path;
    let meta_map = &app.meta;
    let db = app.db.as_ref();
    let classifier_available = app.classifier.is_some();
    let classify_progress = app.classify_progress;
    let rater_available = app.rater.is_some();
    let rate_progress = app.rate_progress;
    let enhancer_available = app.enhancer.is_some();
    let adjust_progress = app.adjust_progress;
    let look_available = app.look.is_some();
    let similar_available =
        app.classifier.is_some() && app.rater.is_some() && app.face_detector.is_some();
    let similar_progress = app.similar_progress;
    let similar_summary = app.similar_summary;
    let show_cull_help = &mut app.flags.show_cull_help;
    let confirm_trash = &mut app.flags.confirm_trash;
    let thumbs = &app.thumbs;


                    egui::Area::new("browse_cull_help".into())
                        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -16.0))
                        .interactable(true)
                        .show(ctx, |ui| {
                            if *show_cull_help {
                                egui::Frame::window(ui.style())
                                    .fill(egui::Color32::from_rgba_premultiplied(24, 24, 24, 238))
                                    .rounding(egui::Rounding::same(8.0))
                                    .inner_margin(egui::Margin::same(10.0))
                                    .show(ui, |ui| {
                                        ui.set_max_width(260.0);
                                        ui.horizontal(|ui| {
                                            ui.label(
                                                egui::RichText::new("Keyboard shortcuts")
                                                    .strong()
                                                    .color(egui::Color32::from_gray(230)),
                                            );
                                            ui.with_layout(
                                                egui::Layout::right_to_left(egui::Align::Center),
                                                |ui| {
                                                    if ui.button("×").clicked() {
                                                        *show_cull_help = false;
                                                        if let Some(db) = db {
                                                            save_bool_pref(db, CULL_HELP_KEY, false);
                                                        }
                                                    }
                                                },
                                            );
                                        });
                                        ui.add_space(4.0);
                                        ui.label("Rate: `1-5`, `0` clear");
                                        ui.label("Flag: `P` pick, `X` reject, `Space` pick+next");
                                        ui.label("Label: `6` red, `7` yellow, `8` green, `9` blue");
                                        ui.label("Move: `←→` / `↑↓`, `Enter` opens");
                                        ui.add_space(4.0);
                                        ui.label(
                                            egui::RichText::new("Bulk actions stay in the top toolbar.")
                                                .small()
                                                .color(egui::Color32::from_gray(150)),
                                        );
                                    });
                            } else if ui.small_button("? Keys").clicked() {
                                *show_cull_help = true;
                                if let Some(db) = db {
                                    save_bool_pref(db, CULL_HELP_KEY, true);
                                }
                            }
                        });

                    egui::SidePanel::left("folder_tree")
                        .default_width(200.0)
                        .width_range(120.0..=380.0)
                        .frame(egui::Frame::none()
                            .fill(egui::Color32::from_gray(18))
                            .inner_margin(egui::Margin::symmetric(4.0, 4.0)))
                        .show(ctx, |ui| {
                            ui.spacing_mut().item_spacing.y = 2.0;
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                let home = std::env::var_os("HOME")
                                    .map(PathBuf::from)
                                    .unwrap_or_default();
                                show_dir_item(ui, &home, 0, tree_expanded, browse_dir.as_deref(), ctx);
                                ui.add_space(4.0);
                                if let Ok(rd) = std::fs::read_dir("/Volumes") {
                                    let mut vols: Vec<PathBuf> = rd
                                        .filter_map(std::result::Result::ok)
                                        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
                                        .map(|e| e.path())
                                        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|s| !s.starts_with('.')))
                                        .collect();
                                    vols.sort_unstable();
                                    for vol in vols {
                                        show_dir_item(ui, &vol, 0, tree_expanded, browse_dir.as_deref(), ctx);
                                    }
                                }
                            });
                        });

                    egui::CentralPanel::default()
                        .frame(egui::Frame::none().fill(egui::Color32::from_gray(24)))
                        .show(ctx, |ui| {
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                ui.add_space(6.0);
                                if ui.button("Open Folder…").clicked() {
                                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                        ctx.data_mut(|d| {
                                            d.insert_temp(egui::Id::new(KEY_SCAN_DIR), dir);
                                        });
                                    }
                                }
                                if let Some(dir) = browse_dir {
                                    ui.label(
                                        egui::RichText::new(dir.display().to_string())
                                            .small()
                                            .color(egui::Color32::from_gray(140)),
                                    );
                                }
                            });
                            ui.add_space(6.0);
                            ui.separator();

                            ui.horizontal_wrapped(|ui| {
                                ui.add_space(6.0);
                                for (f, lbl) in [
                                    (BrowseFilter::All, "All"),
                                    (BrowseFilter::Picks, "Picks"),
                                    (BrowseFilter::Rejects, "Rejects"),
                                    (BrowseFilter::Unflagged, "Unflagged"),
                                ] {
                                    if ui.selectable_label(*filter == f, lbl).clicked() {
                                        *filter = f;
                                    }
                                }
                                ui.separator();
                                ui.label(egui::RichText::new("★ ≥").small());
                                egui::ComboBox::from_id_salt("min_rating")
                                    .selected_text(if *min_rating == 0 { "Any".to_string() } else { "★".repeat(*min_rating as usize) })
                                    .show_ui(ui, |ui| {
                                        for (v, lbl) in [(0u8, "Any"), (1, "★"), (2, "★★"), (3, "★★★"), (4, "★★★★"), (5, "★★★★★")] {
                                            ui.selectable_value(min_rating, v, lbl);
                                        }
                                    });
                                ui.separator();
                                ui.label(egui::RichText::new("Sort").small());
                                for (s, lbl) in [(BrowseSort::Name, "Name"), (BrowseSort::Date, "Date"), (BrowseSort::CaptureTime, "Time")] {
                                    if ui.selectable_label(*sort == s, lbl).clicked() {
                                        *sort = s;
                                    }
                                }
                                ui.separator();
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} shown · {n_picks} picks · {n_rejects} rejects",
                                        visible.len()
                                    ))
                                    .small()
                                    .color(egui::Color32::from_gray(140)),
                                );
                                ui.separator();
                                if ui
                                    .add_enabled(n_rejects > 0, egui::Button::new(format!("Trash Rejects ({n_rejects})")))
                                    .clicked()
                                {
                                    *confirm_trash = true;
                                }
                                if ui
                                    .add_enabled(n_rejects > 0, egui::Button::new("Move Rejects..."))
                                    .clicked()
                                {
                                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_MOVE_REJECTS), dir));
                                    }
                                }
                                if ui
                                    .add_enabled(n_picks > 0, egui::Button::new(format!("Copy Picks ({n_picks})...")))
                                    .clicked()
                                {
                                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_COPY_PICKS), dir));
                                    }
                                }
                                ui.separator();
                                ui.menu_button("🤖 AI Actions", |ui| {
                                    ui.set_min_width(190.0);
                                    if let Some((done, total)) = classify_progress {
                                        ui.label(format!("Tagging {done}/{total}…"));
                                    } else if ui
                                        .add_enabled(
                                            classifier_available && !thumbs.is_empty(),
                                            egui::Button::new("🏷 Tag Folder"),
                                        )
                                        .on_hover_text("Classify every photo in this folder and add tags")
                                        .clicked()
                                    {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_CLASSIFY_FOLDER), true));
                                        ui.close_menu();
                                    }
                                    if let Some((done, total)) = rate_progress {
                                        ui.label(format!("Rating {done}/{total}…"));
                                    } else if ui
                                        .add_enabled(
                                            rater_available && !thumbs.is_empty(),
                                            egui::Button::new("⭐ Rate Folder"),
                                        )
                                        .on_hover_text("Auto-rate every photo in this folder by quality (1-5 stars)")
                                        .clicked()
                                    {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_RATE_FOLDER), true));
                                        ui.close_menu();
                                    }
                                    if let Some((done, total)) = adjust_progress {
                                        ui.label(format!("Adjusting {done}/{total}…"));
                                    } else if ui
                                        .add_enabled(
                                            enhancer_available && !thumbs.is_empty(),
                                            egui::Button::new("✨ Auto Adjust Folder"),
                                        )
                                        .on_hover_text("Apply AI color and tone adjustment to every photo")
                                        .clicked()
                                    {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_ADJUST_FOLDER), true));
                                        ui.close_menu();
                                    }
                                    if ui
                                        .add_enabled(
                                            enhancer_available && look_available && !thumbs.is_empty(),
                                            egui::Button::new("🎨 Apply Look to Folder"),
                                        )
                                        .on_hover_text("Normalize each photo, then apply the captured creative look")
                                        .clicked()
                                    {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_LOOK_FOLDER), true));
                                        ui.close_menu();
                                    }
                                    if let Some((done, total)) = similar_progress {
                                        ui.label(format!("Grouping {done}/{total}…"));
                                    } else if ui
                                        .add_enabled(
                                            similar_available && !thumbs.is_empty(),
                                            egui::Button::new("✨ Cull Similar"),
                                        )
                                        .on_hover_text("Group near-duplicate photos, then keep the best-rated frame; face size breaks ties")
                                        .clicked()
                                    {
                                        ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_SIMILAR_CULL), true));
                                        ui.close_menu();
                                    }
                                });
                                if let Some((done, total)) = classify_progress {
                                    ui.label(
                                        egui::RichText::new(format!("Tagging {done}/{total}…"))
                                            .small()
                                            .color(egui::Color32::from_gray(160)),
                                    );
                                } else if let Some((done, total)) = rate_progress {
                                    ui.label(
                                        egui::RichText::new(format!("Rating {done}/{total}…"))
                                            .small()
                                            .color(egui::Color32::from_gray(160)),
                                    );
                                } else if let Some((done, total)) = adjust_progress {
                                    ui.label(
                                        egui::RichText::new(format!("Adjusting {done}/{total}…"))
                                            .small()
                                            .color(egui::Color32::from_gray(160)),
                                    );
                                } else if let Some((done, total)) = similar_progress {
                                    ui.label(
                                        egui::RichText::new(format!("Grouping {done}/{total}…"))
                                            .small()
                                            .color(egui::Color32::from_gray(160)),
                                    );
                                }
                                if let Some((groups, rejected)) = similar_summary {
                                    ui.label(
                                        egui::RichText::new(format!("{groups} groups · {rejected} rejected"))
                                            .small()
                                            .color(egui::Color32::from_gray(140)),
                                    );
                                }
                            });
                            ui.separator();

                            if thumbs.is_empty() || visible.is_empty() {
                                ui.centered_and_justified(|ui| {
                                    let msg = if thumbs.is_empty() {
                                        "Open a folder (or an image) to browse thumbnails"
                                    } else {
                                        "No images match the filter"
                                    };
                                    ui.label(egui::RichText::new(msg).color(egui::Color32::from_gray(140)));
                                });
                                return;
                            }

                            egui::ScrollArea::vertical().show(ui, |ui| {
                                ui.add_space(6.0);
                                ui.horizontal_wrapped(|ui| {
                                    let cell = egui::vec2(168.0, 152.0);
                                    *grid_cols = (ui.available_width() / cell.x).max(1.0) as usize;
                                    for &ti in visible {
                                        let entry = &thumbs[ti];
                                        let (rect, resp) =
                                            ui.allocate_exact_size(cell, egui::Sense::click());
                                        if *grid_scroll && selected.as_deref() == Some(entry.path.as_path()) {
                                            ui.scroll_to_rect(rect, Some(egui::Align::Center));
                                            *grid_scroll = false;
                                        }
                                        if !ui.is_rect_visible(rect) {
                                            continue;
                                        }
                                        let p = ui.painter_at(rect);
                                        p.rect_filled(rect.shrink(2.0), 4.0, egui::Color32::from_gray(16));

                                        let img_area = egui::Rect::from_min_max(
                                            rect.min + egui::vec2(6.0, 6.0),
                                            egui::pos2(rect.max.x - 6.0, rect.max.y - 36.0),
                                        );
                                        paint_thumb_cell(
                                            &p,
                                            img_area,
                                            entry.tex.as_ref(),
                                            meta_map.get(&entry.path).copied().unwrap_or_default(),
                                            18.0,
                                            true,
                                        );

                                        let name = entry
                                            .path
                                            .file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or("");
                                        let mut label = name.to_owned();
                                        if label.len() > 22 {
                                            label.truncate(20);
                                            label.push('…');
                                        }
                                        // Row 1: filename
                                        p.text(
                                            egui::pos2(rect.center().x, rect.max.y - 24.0),
                                            egui::Align2::CENTER_CENTER,
                                            label,
                                            egui::FontId::proportional(10.0),
                                            egui::Color32::from_gray(170),
                                        );
                                        // Row 2: EXIF (left) + time (right)
                                        let exif_line: String = [
                                            entry.exif.shutter.clone(),
                                            entry.exif.aperture.map(|(n, d)| {
                                                let whole = n / d;
                                                let tenths = (u64::from(n) * 10 / u64::from(d)) % 10;
                                                if tenths < 1 { format!("f/{whole}") } else { format!("f/{whole}.{tenths}") }
                                            }),
                                            entry.exif.iso.map(|i| format!("ISO{i}")),
                                        ].into_iter().flatten().collect::<Vec<_>>().join("  ");
                                        if !exif_line.is_empty() {
                                            p.text(
                                                egui::pos2(rect.min.x + 6.0, rect.max.y - 11.0),
                                                egui::Align2::LEFT_CENTER,
                                                exif_line,
                                                egui::FontId::proportional(9.0),
                                                egui::Color32::from_gray(120),
                                            );
                                        }
                                        if let Some(ct) = entry.exif.capture_time {
                                            p.text(
                                                egui::pos2(rect.max.x - 6.0, rect.max.y - 11.0),
                                                egui::Align2::RIGHT_CENTER,
                                                format!("{}-{:02}-{:02} {:02}:{:02}", ct / 10_000_000_000, (ct / 100_000_000) % 100, (ct / 1_000_000) % 100, (ct / 10_000) % 100, (ct / 100) % 100),
                                                egui::FontId::proportional(9.0),
                                                egui::Color32::from_gray(120),
                                            );
                                        }

                                        let selected_item = selected
                                            .as_deref()
                                            .or(current_path.as_deref())
                                            == Some(entry.path.as_path());
                                        if selected_item {
                                            p.rect_stroke(
                                                rect.shrink(2.0),
                                                4.0,
                                                egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(90, 140, 255)),
                                            );
                                        } else if resp.hovered() {
                                            p.rect_stroke(
                                                rect.shrink(2.0),
                                                4.0,
                                                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(110)),
                                            );
                                        }

                                        if resp.clicked() {
                                            *selected = Some(entry.path.clone());
                                        }
                                        if resp.double_clicked() {
                                            *selected = Some(entry.path.clone());
                                            ctx.data_mut(|d| {
                                                d.insert_temp(
                                                    egui::Id::new(KEY_OPEN_PATH),
                                                    entry.path.clone(),
                                                );
                                            });
                                        }
                                    }
                                });
                                ui.add_space(6.0);
                            });
                        });

}

// ---- Edit view -------------------------------------------------------------

/// Brush radius in image pixels, clamped to what the search can work with:
/// at least two pixels, and no more than a quarter of the shorter side.
fn brush_radius_px(min_dim: f32, radius: f32) -> f32 {
    radius.clamp(2.0, (min_dim * 0.25).max(2.0))
}

/// How far the image moves this frame. A drag and a two-finger trackpad swipe
/// are both reported by egui in "how the content should move" units, so either
/// one just adds to the offset; a drag wins if both land in the same frame.
/// Neither moves anything while the image is fitted, where there is nothing to
/// pan to.
fn pan_delta(fit: bool, drag: Option<egui::Vec2>, scroll: egui::Vec2) -> egui::Vec2 {
    if fit {
        return egui::Vec2::ZERO;
    }
    drag.unwrap_or(scroll)
}

/// Offset that keeps the image point under `anchor` (panel coordinates) fixed
/// while the scale changes from `old_scale` to `new_scale`. Shared by the wheel
/// and by a trackpad pinch, which must both scale *about the cursor*.
fn zoom_about(anchor: egui::Vec2, old_scale: f32, new_scale: f32, offset: egui::Vec2) -> egui::Vec2 {
    anchor - (anchor - offset) * (new_scale / old_scale)
}

/// Step the brush radius by `steps` (`[` is -1, `]` is +1). Multiplicative, so
/// the feel is the same at every size, and clamped like [`brush_radius_px`].
fn step_brush(min_dim: f32, radius: f32, steps: i32) -> f32 {
    let mut r = radius;
    for _ in 0..steps.unsigned_abs() {
        r = if steps < 0 { r * 0.8 } else { r * 1.25 };
    }
    brush_radius_px(min_dim, r)
}

/// Crop aspect presets, by index. 0 is free-form, 1 is the image's own aspect;
/// the rest are fixed ratios.
const CROP_ASPECTS: [&str; 8] = ["Free", "Original", "1:1", "4:5", "5:4", "3:2", "2:3", "16:9"];

/// The width/height ratio a preset locks to, if any.
fn aspect_ratio(index: usize, input: (f32, f32)) -> Option<f32> {
    match index {
        0 => None,
        1 => Some(input.0 / input.1),
        2 => Some(1.0),
        3 => Some(4.0 / 5.0),
        4 => Some(5.0 / 4.0),
        5 => Some(3.0 / 2.0),
        6 => Some(2.0 / 3.0),
        _ => Some(16.0 / 9.0),
    }
}

/// Smallest crop, in image pixels: below this the handles overlap and the
/// result is useless.
const MIN_CROP_PX: f32 = 16.0;

/// A crop rectangle as `(x0, y0, x1, y1)` in image pixels.
type CropPx = (f32, f32, f32, f32);

/// Reshape a crop to a fixed width/height ratio about its centre, shrinking it
/// if the image cannot hold that shape at the current size.
fn crop_with_aspect(rect: CropPx, input: (f32, f32), ratio: f32) -> CropPx {
    let (cx, cy) = ((rect.0 + rect.2) * 0.5, (rect.1 + rect.3) * 0.5);
    let (mut w, mut h) = (rect.2 - rect.0, rect.3 - rect.1);
    if w / h > ratio {
        w = h * ratio;
    } else {
        h = w / ratio;
    }
    crop_inside((cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), input)
}

/// Shrink and slide a crop until it sits inside the image at a usable size,
/// keeping its centre where it can.
fn crop_inside(rect: CropPx, input: (f32, f32)) -> CropPx {
    let w = (rect.2 - rect.0).clamp(MIN_CROP_PX.min(input.0), input.0);
    let h = (rect.3 - rect.1).clamp(MIN_CROP_PX.min(input.1), input.1);
    let cx = ((rect.0 + rect.2) * 0.5).clamp(w * 0.5, input.0 - w * 0.5);
    let cy = ((rect.1 + rect.3) * 0.5).clamp(h * 0.5, input.1 - h * 0.5);
    (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)
}

/// Where a drag leaves the crop. `sides` names the edges the pointer drags
/// (bit 0 left, 1 right, 2 top, 3 bottom; zero moves the whole rectangle).
/// With an aspect lock the rectangle is reshaped about its centre, the
/// dominant finger axis deciding which way; it never crosses over itself, drops
/// below [`MIN_CROP_PX`], or leaves the image.
fn drag_crop(
    rect: CropPx,
    input: (f32, f32),
    sides: u8,
    delta: egui::Vec2,
    aspect: Option<f32>,
) -> CropPx {
    const LEFT: u8 = 1;
    const RIGHT: u8 = 2;
    const TOP: u8 = 4;
    const BOTTOM: u8 = 8;
    let (mut x0, mut y0, mut x1, mut y1) = rect;
    if sides == 0 {
        return crop_inside(
            (x0 + delta.x, y0 + delta.y, x1 + delta.x, y1 + delta.y),
            input,
        );
    }
    let min_w = MIN_CROP_PX.min(input.0);
    let min_h = MIN_CROP_PX.min(input.1);
    if sides & LEFT != 0 {
        x0 = (x0 + delta.x).min(x1 - min_w);
    }
    if sides & RIGHT != 0 {
        x1 = (x1 + delta.x).max(x0 + min_w);
    }
    if sides & TOP != 0 {
        y0 = (y0 + delta.y).min(y1 - min_h);
    }
    if sides & BOTTOM != 0 {
        y1 = (y1 + delta.y).max(y0 + min_h);
    }
    if let Some(ratio) = aspect {
        let (cx, cy) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        let (mut w, mut h) = (x1 - x0, y1 - y0);
        let horizontal = sides & (LEFT | RIGHT) != 0;
        let vertical = sides & (TOP | BOTTOM) != 0;
        let width_drives = if horizontal && vertical {
            delta.x.abs() / w.max(1.0) >= delta.y.abs() / h.max(1.0)
        } else {
            horizontal
        };
        if width_drives {
            h = w / ratio;
        } else {
            w = h * ratio;
        }
        x0 = cx - w * 0.5;
        x1 = cx + w * 0.5;
        y0 = cy - h * 0.5;
        y1 = cy + h * 0.5;
    }
    crop_inside((x0, y0, x1, y1), input)
}

/// Maps between image coordinates (fractions of the input image) and the
/// screen, through whatever crop the panel is showing. Everything that has to
/// place something on the image goes through here, so a crop carries the
/// retouch markers with the pixels instead of leaving them behind.
#[derive(Clone, Copy)]
struct ImageView {
    /// Screen rectangle the displayed image occupies.
    rect: egui::Rect,
    /// Which part of the input image it shows, as `(x, y, w, h)` fractions.
    uv: [f32; 4],
    /// Input image size, in pixels.
    input: (f32, f32),
}

impl ImageView {
    fn to_screen(self, x: f32, y: f32) -> egui::Pos2 {
        egui::pos2(
            self.rect.min.x + (x - self.uv[0]) / self.uv[2] * self.rect.width(),
            self.rect.min.y + (y - self.uv[1]) / self.uv[3] * self.rect.height(),
        )
    }

    fn to_input(self, p: egui::Pos2) -> [f32; 2] {
        [
            self.uv[0] + (p.x - self.rect.min.x) / self.rect.width() * self.uv[2],
            self.uv[1] + (p.y - self.rect.min.y) / self.rect.height() * self.uv[3],
        ]
    }

    /// Screen length of one input-image pixel. The fit preserves the aspect, so
    /// the two axes agree; x is used for both.
    fn px_scale(&self) -> f32 {
        self.rect.width() / (self.uv[2] * self.input.0)
    }
}

/// What the pointer is doing to the crop rectangle.
#[derive(Clone, Copy)]
pub(crate) struct CropDrag {
    /// Edges the pointer drags: bit 0 left, 1 right, 2 top, 3 bottom. Zero
    /// moves the whole rectangle.
    pub(crate) sides: u8,
}

/// Which edges of the crop a pointer near `p` grabs: bit 0 left, 1 right,
/// 2 top, 3 bottom, zero to move the whole rectangle. `None` when the pointer is
/// away from the rectangle, so a drag elsewhere leaves it alone.
fn crop_handle(rect: egui::Rect, p: egui::Pos2, reach: f32) -> Option<u8> {
    if p.x < rect.left() - reach
        || p.x > rect.right() + reach
        || p.y < rect.top() - reach
        || p.y > rect.bottom() + reach
    {
        return None;
    }
    let mut sides = 0u8;
    if (p.x - rect.left()).abs() <= reach {
        sides |= 1;
    }
    if (p.x - rect.right()).abs() <= reach {
        sides |= 2;
    }
    if (p.y - rect.top()).abs() <= reach {
        sides |= 4;
    }
    if (p.y - rect.bottom()).abs() <= reach {
        sides |= 8;
    }
    Some(sides)
}

/// The crop tool's layer: the rectangle with everything outside it dimmed, its
/// handles, and the drag that resizes it. It is drawn against the *whole*
/// frame, which is what you crop against. Returns the new crop when the drag
/// moved it.
#[allow(clippy::too_many_arguments)]
fn paint_crop_tool(
    ui: &egui::Ui,
    panel: egui::Rect,
    view: &ImageView,
    crop: CropPx,
    input: (f32, f32),
    aspect: Option<f32>,
    drag: &mut Option<CropDrag>,
    response: &egui::Response,
    ctx: &egui::Context,
) -> Option<CropPx> {
    let painter = ui.painter().with_clip_rect(panel);
    let rect = egui::Rect::from_min_max(
        view.to_screen(crop.0 / input.0, crop.1 / input.1),
        view.to_screen(crop.2 / input.0, crop.3 / input.1),
    );

    // Everything outside the crop goes dark.
    let dim = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 140);
    for band in [
        egui::Rect::from_min_max(panel.min, egui::pos2(panel.right(), rect.top())),
        egui::Rect::from_min_max(egui::pos2(panel.left(), rect.bottom()), panel.max),
        egui::Rect::from_min_max(
            egui::pos2(panel.left(), rect.top()),
            egui::pos2(rect.left(), rect.bottom()),
        ),
        egui::Rect::from_min_max(
            egui::pos2(rect.right(), rect.top()),
            egui::pos2(panel.right(), rect.bottom()),
        ),
    ] {
        painter.rect_filled(band, 0.0, dim);
    }

    // Rule of thirds, so the crop is easy to place.
    let guide = egui::Stroke::new(1.0_f32, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 60));
    for i in 1..3 {
        let t = i as f32 / 3.0;
        let x = rect.left() + rect.width() * t;
        let y = rect.top() + rect.height() * t;
        painter.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())], guide);
        painter.line_segment([egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)], guide);
    }
    painter.rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(
            if drag.is_some() { 2.0_f32 } else { 1.0_f32 },
            egui::Color32::WHITE,
        ),
    );
    let handle = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 220);
    for (hx, hy) in [
        (rect.left(), rect.top()),
        (rect.center().x, rect.top()),
        (rect.right(), rect.top()),
        (rect.left(), rect.center().y),
        (rect.right(), rect.center().y),
        (rect.left(), rect.bottom()),
        (rect.center().x, rect.bottom()),
        (rect.right(), rect.bottom()),
    ] {
        painter.rect_filled(
            egui::Rect::from_center_size(egui::pos2(hx, hy), egui::vec2(7.0, 7.0)),
            1.0,
            handle,
        );
    }

    // Grab reach shrinks with the rectangle, so a small crop can still be moved
    // rather than being all handle.
    let reach = 10.0_f32.min(rect.width() / 3.0).min(rect.height() / 3.0).max(2.0);
    let grabbed = response.hover_pos().and_then(|p| crop_handle(rect, p, reach));
    if drag.is_none() {
        if let Some(sides) = grabbed {
            let horizontal = sides & (1 | 2) != 0;
            let vertical = sides & (4 | 8) != 0;
            ctx.set_cursor_icon(match (horizontal, vertical) {
                (true, true) => egui::CursorIcon::ResizeNwSe,
                (true, false) => egui::CursorIcon::ResizeHorizontal,
                (false, true) => egui::CursorIcon::ResizeVertical,
                (false, false) => egui::CursorIcon::Move,
            });
        }
    }

    if response.drag_started() {
        *drag = grabbed.map(|sides| CropDrag { sides });
    }
    let mut moved = None;
    if response.dragged() {
        if let Some(state) = *drag {
            // The pointer delta is in screen points; the crop lives in pixels.
            let scale = view.px_scale().max(1e-6);
            moved = Some(drag_crop(
                crop,
                input,
                state.sides,
                response.drag_delta() / scale,
                aspect,
            ));
        }
    }
    if response.drag_stopped() {
        *drag = None;
    }
    moved
}

/// The retouch tool's on-screen layer: a marker ring for every dab, the brush
/// cursor under the pointer, and the clicks that place or delete a dab.
///
/// Markers only exist while the tool is armed -- the dabs themselves live in
/// the edit state, applied on the GPU, and stay once the tool is put away.
/// Everything is placed through the [`ImageView`], so a crop carries the
/// markers with the pixels.
fn paint_retouch(
    ui: &egui::Ui,
    panel: egui::Rect,
    view: &ImageView,
    brush_radius: f32,
    response: &egui::Response,
    ctx: &egui::Context,
    spots: &[processor::RetouchSpot],
) {
    let painter = ui.painter().with_clip_rect(panel);
    // A dab's radius is a fraction of the input image's shorter side.
    let short = view.input.0.min(view.input.1);
    let ring = |spot: &processor::RetouchSpot| {
        (
            view.to_screen(spot.x, spot.y),
            (spot.radius * short * view.px_scale()).max(3.0),
        )
    };
    let brush_screen = (brush_radius * view.px_scale()).max(2.0);
    let pointer = response.hover_pos();

    // The dab a click acts on is the last one drawn there -- the one on top,
    // which is what the markers show.
    let topmost = pointer.and_then(|p| {
        spots
            .iter()
            .enumerate()
            .filter(|(_, spot)| {
                let (c, r) = ring(spot);
                (p - c).length() <= r.max(6.0)
            })
            .map(|(index, _)| index)
            .next_back()
    });

    for (index, spot) in spots.iter().enumerate() {
        let (c, r) = ring(spot);
        painter.circle_filled(c, r, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 24));
        painter.add(egui::Shape::circle_stroke(
            c,
            r,
            if Some(index) == topmost {
                egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(255, 132, 120))
            } else {
                egui::Stroke::new(1.0_f32, egui::Color32::WHITE)
            },
        ));
        // A dark companion ring keeps the marker readable over bright pixels.
        painter.add(egui::Shape::circle_stroke(
            c,
            r + 1.0,
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120)),
        ));
    }

    let Some(hover) = pointer.filter(|p| view.rect.contains(*p)) else {
        return;
    };

    // Brush cursor: a white ring with a dark companion and a small crosshair.
    let dark = egui::Stroke::new(2.0_f32, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 150));
    let light = egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
    painter.add(egui::Shape::circle_stroke(hover, brush_screen + 1.0, dark));
    painter.add(egui::Shape::circle_stroke(hover, brush_screen, light));
    painter.line_segment([hover - egui::vec2(4.0, 0.0), hover + egui::vec2(4.0, 0.0)], light);
    painter.line_segment([hover - egui::vec2(0.0, 4.0), hover + egui::vec2(0.0, 4.0)], light);

    // Left click heals the spot under the cursor. The click arrives in the
    // displayed (possibly cropped) frame and is stored against the whole image.
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos().filter(|p| view.rect.contains(*p)) {
            let [x, y] = view.to_input(pos);
            if (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y) {
                let radius = (brush_radius / short).clamp(0.0015, 0.25);
                ctx.data_mut(|d| {
                    d.insert_temp(egui::Id::new(KEY_RETOUCH), [x, y, radius]);
                });
            }
        }
    }

    // Right click deletes the dab on top under the cursor.
    if response.secondary_clicked() {
        if let Some(index) = topmost {
            ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_RETOUCH_REMOVE), index));
        }
    }
}

/// Brush-size step for one key event: -1 for `[`, +1 for `]`, 0 otherwise.
///
/// Matched on the **physical** key rather than the character it types: on the
/// Czech layout `[` and `]` sit under `ú` and `)`, so a character match would
/// miss them. The physical key is the same on every layout.
fn brush_size_step(key: &egui::Key, physical_key: &Option<egui::Key>) -> i32 {
    let is = |wanted: egui::Key| *key == wanted || *physical_key == Some(wanted);
    if is(egui::Key::OpenBracket) {
        -1
    } else if is(egui::Key::CloseBracket) {
        1
    } else {
        0
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn edit(
    ctx: &egui::Context,
    app: &mut App,
    cull_actions: &mut Vec<(PathBuf, cull::CullAction)>,
    needs_process: &mut bool,
    visible: &[usize],
) {
    let processor = app.processor.as_mut().unwrap();
    let zoom_fit = &mut app.flags.zoom_fit;
    let zoom_scale = &mut app.zoom_scale;
    let zoom_offset = &mut app.zoom_offset;
    let preview_hold_start = &mut app.preview_hold_start;
    let levels_drag = &mut app.levels_drag;
    let curve_drag = &mut app.curve_drag;
    let selected = &mut app.selected;
    let strip_scroll = &mut app.flags.strip_scroll;
    let current_path = &app.current_path;
    let meta_map = &app.meta;
    let db = app.db.as_ref();
    let presets_dir = app.presets_dir.as_deref();
    let presets = &mut app.presets;
    let preset_name = &mut app.preset_name;
    let look_available = app.look.is_some();
    let current_is_raw = current_path.as_deref().is_some_and(imgload::is_raw);
    let tags_map = &mut app.tags;
    let tag_edit = &mut app.tag_edit;
    let retouch_active = &mut app.retouch_active;
    let brush_radius = &mut app.brush_radius;
    let retouch_status = &mut app.retouch_status;
    let retouch_confirm_clear = &mut app.retouch_confirm_clear;
    let crop_active = &mut app.crop_active;
    let crop_aspect = &mut app.crop_aspect;
    let crop_drag = &mut app.crop_drag;
    let export_progress = &app.export_progress;
    let thumbs = &app.thumbs;
    let view = &app.view;
    let image_tex_id = app.image_tex_id;
    let original_tex_id = app.original_tex_id;
    let reference_tex_id = app.reference_tex.as_ref().map(egui::TextureHandle::id);
    let reference_size = app
        .look
        .as_ref()
        .map(|look| (look.reference_full.width(), look.reference_full.height()));


                egui::SidePanel::right("controls")
                    .exact_width(260.0)
                    .resizable(false)
                    .show(ctx, |ui| {
                        egui::ScrollArea::vertical()
                            .id_salt("editor-controls-scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let resp = ui
                                .selectable_label(*retouch_active, "⚕ Retouch")
                                .on_hover_text(
                                    "Healing brush: click the photo to heal a spot.\n\
                                     [ and ] change the size · right-click a dab to delete it",
                                );
                            if resp.clicked() {
                                *retouch_active = !*retouch_active;
                                *retouch_confirm_clear = false;
                                // One tool owns the pointer at a time.
                                *crop_active = false;
                            }
                            if *retouch_active {
                                ui.label(
                                    egui::RichText::new(format!("⌀ {:.0} px", *brush_radius * 2.0))
                                        .small()
                                        .color(egui::Color32::from_gray(150)),
                                );
                            }
                        });
                        if *retouch_active {
                            ui.label(
                                egui::RichText::new("Click to heal · [ / ] size")
                                    .small()
                                    .color(egui::Color32::from_gray(130)),
                            );
                        }
                        // A click that placed nothing (usually no source inside
                        // the image) says so, briefly, instead of doing nothing.
                        if let Some((message, raised)) = retouch_status.as_ref() {
                            if ctx.input(|i| i.time) - raised < 6.0 {
                                ui.label(
                                    egui::RichText::new(message.as_str())
                                        .small()
                                        .color(egui::Color32::from_rgb(230, 170, 90)),
                                );
                            }
                        }
                        let spots = processor.retouch_spots.len();
                        if spots > 0 {
                            if spots >= processor::MAX_RETOUCH_SPOTS {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Retouch limit reached ({})",
                                        processor::MAX_RETOUCH_SPOTS
                                    ))
                                    .small()
                                    .color(egui::Color32::from_rgb(220, 170, 90)),
                                );
                            }
                            // Clearing every dab cannot be undone, so it takes
                            // two clicks.
                            if !*retouch_confirm_clear {
                                if ui.button(format!("Clear {spots} retouch(es)")).clicked() {
                                    *retouch_confirm_clear = true;
                                }
                            } else {
                                ui.horizontal(|ui| {
                                    if ui
                                        .button(
                                            egui::RichText::new("Really clear all")
                                                .color(egui::Color32::from_rgb(235, 120, 110)),
                                        )
                                        .clicked()
                                    {
                                        *retouch_confirm_clear = false;
                                        ctx.data_mut(|d| {
                                            d.insert_temp(egui::Id::new(KEY_RETOUCH_CLEAR), true)
                                        });
                                    }
                                    if ui.button("Cancel").clicked() {
                                        *retouch_confirm_clear = false;
                                    }
                                });
                            }
                        }
                        let crop_input = processor.image_size.unwrap_or((1, 1));
                        let crop_ratio = aspect_ratio(*crop_aspect, (crop_input.0 as f32, crop_input.1 as f32));
                        ui.horizontal(|ui| {
                            let resp = ui
                                .selectable_label(*crop_active, "⛶ Crop")
                                .on_hover_text(
                                    "Crop the frame: drag the corners or edges, or inside \
                                     to move it. Non-destructive.",
                                );
                            if resp.clicked() {
                                *crop_active = !*crop_active;
                                // One tool owns the pointer at a time, and a
                                // crop is placed against the whole frame.
                                *retouch_active = false;
                                if *crop_active {
                                    *zoom_fit = true;
                                }
                            }
                            let (_, _, cw, ch) = processor::crop_px(processor.crop, crop_input);
                            if !processor::is_full_crop(processor.crop, crop_input) {
                                ui.label(
                                    egui::RichText::new(format!("{cw}×{ch}"))
                                        .small()
                                        .color(egui::Color32::from_gray(150)),
                                );
                            }
                        });
                        if *crop_active {
                            let mut preset = None;
                            ui.horizontal_wrapped(|ui| {
                                for (index, label) in CROP_ASPECTS.iter().enumerate() {
                                    if ui
                                        .selectable_value(crop_aspect, index, *label)
                                        .changed()
                                    {
                                        preset = Some(index);
                                    }
                                }
                            });
                            // Locking a ratio reshapes the crop straight away,
                            // so the setting is visible without touching a handle.
                            if let (Some(index), Some(ratio)) = (preset, crop_ratio) {
                                let (x, y, w, h) = processor::crop_px(processor.crop, crop_input);
                                let rect = (x as f32, y as f32, (x + w) as f32, (y + h) as f32);
                                let reshaped = crop_with_aspect(
                                    rect,
                                    (crop_input.0 as f32, crop_input.1 as f32),
                                    ratio,
                                );
                                processor.crop =
                                    processor::crop_from_px(reshaped, crop_input);
                                *needs_process = true;
                                let _ = index;
                            }
                            if ui.button("Reset crop").clicked() {
                                processor.crop = processor::FULL_CROP;
                                *needs_process = true;
                            }
                        }
                        ui.separator();
                        egui::CollapsingHeader::new("Camera")
                            .default_open(true)
                            .show(ui, |ui| {
                                egui::ScrollArea::horizontal().show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        for category in camera_presets::CATEGORIES {
                                            ui.selectable_value(
                                                &mut app.camera_category,
                                                category,
                                                category.label(),
                                            );
                                        }
                                    });
                                });
                                ui.horizontal_wrapped(|ui| {
                                    let selected = processor.camera_preset.is_none();
                                    if ui.selectable_label(selected, "None").clicked() {
                                        processor.camera_preset = None;
                                        processor.camera_style = 0;
                                        *needs_process = true;
                                    }
                                    for preset in app.camera_category.presets() {
                                        let selected = processor.camera_preset == Some(preset);
                                        if ui.selectable_label(selected, preset.name()).clicked() {
                                            processor.camera_preset = Some(preset);
                                            processor.camera_style = preset.default_style();
                                            *needs_process = true;
                                        }
                                    }
                                });
                                if let Some(preset) = processor.camera_preset {
                                    let styles = preset.styles();
                                    if !styles.is_empty() {
                                        let selected = styles
                                            .get(processor.camera_style as usize)
                                            .copied()
                                            .unwrap_or(styles[0]);
                                        egui::ComboBox::from_id_salt("camera-style")
                                            .selected_text(selected)
                                            .show_ui(ui, |ui| {
                                                for (index, style) in styles.iter().enumerate() {
                                                    if ui.selectable_label(selected == *style, *style).clicked() {
                                                        processor.camera_style = index as u8;
                                                        *needs_process = true;
                                                    }
                                                }
                                            });
                                        if ui.small_button("Reset style").clicked() {
                                            processor.camera_style = preset.default_style();
                                            *needs_process = true;
                                        }
                                    }
                                    let mut strength_percent = processor.camera_strength * 100.0;
                                    if ui
                                        .add(
                                            egui::Slider::new(&mut strength_percent, 0.0..=200.0)
                                                .text("Strength")
                                                .suffix("%"),
                                        )
                                        .changed()
                                    {
                                        processor.camera_strength = strength_percent / 100.0;
                                        *needs_process = true;
                                    }
                                }
                            });
                        ui.separator();
                        const MIN_DX: f32 = 0.01;
                        const GAMMA_LOG_RANGE: f32 = 1.609_438; // ln(5): gamma handle maps 0.2..5
                        // Luminance histogram + curves editor
                        let hist_size = egui::vec2(ui.available_width(), 150.0);
                        let (hist_rect, hist_resp) =
                            ui.allocate_exact_size(hist_size, egui::Sense::click_and_drag());
                        let painter = ui.painter_at(hist_rect);
                        painter.rect_filled(hist_rect, 0.0, egui::Color32::from_gray(12));

                        if processor.has_image() {
                            let hist = &processor.histogram;
                            let max = hist.iter().copied().max().unwrap_or(1).max(1) as f32;
                            let log_max = (max + 1.0).ln().max(1.0);
                            let bar_w  = hist_rect.width() / 256.0;
                            let b      = hist_rect.bottom();
                            let color  = egui::Color32::from_gray(190);
                            let uv     = egui::epaint::WHITE_UV;

                            let hs: Vec<f32> = hist.iter().map(|&c| {
                                ((c as f32 + 1.0).ln() / log_max) * hist_rect.height()
                            }).collect();
                            let cx = |i: usize| hist_rect.left() + (i as f32 + 0.5) * bar_w;

                            // Explicit mesh: left cap + 255 trapezoids + right cap.
                            // Bypasses PathShape tessellation so fill is always correct.
                            let mut mesh = egui::Mesh::default();
                            let push3 = |m: &mut egui::Mesh, p: [egui::Pos2; 3]| {
                                let v = m.vertices.len() as u32;
                                for pos in p { m.vertices.push(egui::epaint::Vertex { pos, uv, color }); }
                                m.indices.extend_from_slice(&[v, v+1, v+2]);
                            };
                            let push4 = |m: &mut egui::Mesh, p: [egui::Pos2; 4]| {
                                let v = m.vertices.len() as u32;
                                for pos in p { m.vertices.push(egui::epaint::Vertex { pos, uv, color }); }
                                m.indices.extend_from_slice(&[v, v+1, v+2, v, v+2, v+3]);
                            };

                            // Left cap triangle
                            push3(&mut mesh, [
                                egui::pos2(hist_rect.left(), b),
                                egui::pos2(cx(0), b - hs[0]),
                                egui::pos2(cx(0), b),
                            ]);
                            // Trapezoids between adjacent bin centers
                            for i in 0..255 {
                                push4(&mut mesh, [
                                    egui::pos2(cx(i),   b - hs[i]),
                                    egui::pos2(cx(i+1), b - hs[i+1]),
                                    egui::pos2(cx(i+1), b),
                                    egui::pos2(cx(i),   b),
                                ]);
                            }
                            // Right cap triangle
                            push3(&mut mesh, [
                                egui::pos2(cx(255), b - hs[255]),
                                egui::pos2(hist_rect.right(), b),
                                egui::pos2(cx(255), b),
                            ]);

                            painter.add(egui::Shape::Mesh(mesh));
                        }

                        // ---- Curves editor (Photoshop-style) overlaid on the histogram ----
                        let to_screen = |p: [f32; 2]| {
                            egui::pos2(
                                hist_rect.left() + p[0] * hist_rect.width(),
                                hist_rect.bottom() - p[1] * hist_rect.height(),
                            )
                        };
                        let to_norm = |p: egui::Pos2| {
                            [
                                ((p.x - hist_rect.left()) / hist_rect.width()).clamp(0.0, 1.0),
                                ((hist_rect.bottom() - p.y) / hist_rect.height()).clamp(0.0, 1.0),
                            ]
                        };
                        let nearest_point = |pts: &[[f32; 2]], mp: egui::Pos2| {
                            pts.iter()
                                .enumerate()
                                .map(|(i, &p)| (i, to_screen(p).distance(mp)))
                                .min_by(|a, b| a.1.total_cmp(&b.1))
                                .filter(|(_, d)| *d < 10.0)
                                .map(|(i, _)| i)
                        };

                        // Grab an existing point or create one (click or drag start)
                        if hist_resp.drag_started() || hist_resp.clicked() {
                            if let Some(mp) = hist_resp.interact_pointer_pos() {
                                let pts = &mut processor.curve_points;
                                let idx = nearest_point(pts, mp).unwrap_or_else(|| {
                                    let np = to_norm(mp);
                                    let idx = pts.iter().position(|p| p[0] > np[0]).unwrap_or(pts.len());
                                    pts.insert(idx, np);
                                    *needs_process = true;
                                    idx
                                });
                                *curve_drag = Some(idx);
                            }
                        }
                        if hist_resp.dragged() {
                            if let (Some(i), Some(mp)) = (*curve_drag, hist_resp.interact_pointer_pos()) {
                                let pts = &mut processor.curve_points;
                                let np = to_norm(mp);
                                let last = pts.len() - 1;
                                let x = if i == 0 {
                                    np[0].min(pts[1][0] - MIN_DX).max(0.0)
                                } else if i == last {
                                    np[0].max(pts[last - 1][0] + MIN_DX).min(1.0)
                                } else {
                                    np[0].clamp(pts[i - 1][0] + MIN_DX, pts[i + 1][0] - MIN_DX)
                                };
                                pts[i] = [x, np[1]];
                                *needs_process = true;
                            }
                        }
                        if hist_resp.drag_stopped() {
                            // Releasing a point well outside the box deletes it (endpoints stay)
                            if let (Some(i), Some(mp)) = (*curve_drag, ctx.pointer_interact_pos()) {
                                let pts = &mut processor.curve_points;
                                if i != 0 && i != pts.len() - 1 && !hist_rect.expand(25.0).contains(mp) {
                                    pts.remove(i);
                                    *needs_process = true;
                                }
                            }
                            *curve_drag = None;
                        }
                        // Right-click removes a point; double-click resets the whole curve
                        if hist_resp.secondary_clicked() {
                            if let Some(mp) = hist_resp.interact_pointer_pos() {
                                let pts = &mut processor.curve_points;
                                if let Some(i) = nearest_point(pts, mp) {
                                    if i != 0 && i != pts.len() - 1 {
                                        pts.remove(i);
                                        *needs_process = true;
                                    }
                                }
                            }
                        }
                        if hist_resp.double_clicked() {
                            processor.curve_points = vec![[0.0, 0.0], [1.0, 1.0]];
                            *curve_drag = None;
                            *needs_process = true;
                        }

                        // Quarter grid lines
                        let grid_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_gray(34));
                        for f in [0.25f32, 0.5, 0.75] {
                            let x = hist_rect.left() + f * hist_rect.width();
                            let y = hist_rect.top() + f * hist_rect.height();
                            painter.line_segment([egui::pos2(x, hist_rect.top()), egui::pos2(x, hist_rect.bottom())], grid_stroke);
                            painter.line_segment([egui::pos2(hist_rect.left(), y), egui::pos2(hist_rect.right(), y)], grid_stroke);
                        }

                        // The curve itself, sampled from the same LUT the shader uses
                        let lut = processor.curve_lut();
                        let curve_pts: Vec<egui::Pos2> = (0..=128)
                            .map(|i| {
                                let t = i as f32 / 128.0;
                                to_screen([t, lut[((t * 255.0) as usize).min(255)]])
                            })
                            .collect();
                        painter.add(egui::Shape::line(
                            curve_pts,
                            egui::Stroke::new(1.5_f32, egui::Color32::from_gray(230)),
                        ));

                        // Control points
                        let hover_idx = hist_resp
                            .hover_pos()
                            .and_then(|mp| nearest_point(&processor.curve_points, mp));
                        for (i, &p) in processor.curve_points.iter().enumerate() {
                            let hot = *curve_drag == Some(i) || (curve_drag.is_none() && hover_idx == Some(i));
                            let (r, fill) = if hot {
                                (4.5, egui::Color32::WHITE)
                            } else {
                                (3.5, egui::Color32::from_gray(200))
                            };
                            painter.circle(to_screen(p), r, fill, egui::Stroke::new(1.0_f32, egui::Color32::from_gray(60)));
                        }
                        if hover_idx.is_some() || curve_drag.is_some() {
                            ctx.set_cursor_icon(egui::CursorIcon::Grab);
                        }

                        // Levels: gradient strip with draggable handles (black / gamma / white)
                        let strip_h  = 10.0;
                        let marker_h = 12.0;
                        let (strip_area, strip_resp) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), strip_h + marker_h),
                            egui::Sense::click_and_drag(),
                        );
                        let sp = ui.painter_at(strip_area);
                        let gradient_rect = egui::Rect::from_min_size(strip_area.min, egui::vec2(strip_area.width(), strip_h));

                        let w = strip_area.width();
                        let x_of = |v: f32| strip_area.left() + (v / 255.0) * w;
                        let black_x = x_of(processor.levels_black);
                        let white_x = x_of(processor.levels_white);
                        // Gamma handle position between black/white, log-symmetric so center = 1.0
                        let gamma_t = (0.5 - processor.levels_gamma.ln() / (2.0 * GAMMA_LOG_RANGE)).clamp(0.0, 1.0);
                        let gamma_x = black_x + gamma_t * (white_x - black_x);

                        // Interaction: grab nearest handle on drag start, follow pointer while dragging
                        if strip_resp.drag_started() {
                            if let Some(p) = strip_resp.interact_pointer_pos() {
                                let dists = [(p.x - black_x).abs(), (p.x - gamma_x).abs(), (p.x - white_x).abs()];
                                *levels_drag = dists
                                    .iter()
                                    .enumerate()
                                    .min_by(|a, b| a.1.total_cmp(b.1))
                                    .map(|(i, _)| i);
                            }
                        }
                        if strip_resp.dragged() {
                            if let (Some(h), Some(p)) = (*levels_drag, strip_resp.interact_pointer_pos()) {
                                let v = ((p.x - strip_area.left()) / w * 255.0).clamp(0.0, 255.0);
                                match h {
                                    0 => processor.levels_black = v.round().clamp(0.0, processor.levels_white - 1.0),
                                    2 => processor.levels_white = v.round().clamp(processor.levels_black + 1.0, 255.0),
                                    _ => {
                                        let t = ((p.x - black_x) / (white_x - black_x).max(1.0)).clamp(0.0, 1.0);
                                        processor.levels_gamma = ((0.5 - t) * 2.0 * GAMMA_LOG_RANGE).exp();
                                    }
                                }
                                *needs_process = true;
                            }
                        }
                        if strip_resp.drag_stopped() {
                            *levels_drag = None;
                        }
                        if strip_resp.double_clicked() {
                            processor.levels_black = 0.0;
                            processor.levels_white = 255.0;
                            processor.levels_gamma = 1.0;
                            *needs_process = true;
                        }
                        if strip_resp.hovered() || levels_drag.is_some() {
                            ctx.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                        }

                        // Black → white gradient
                        let uv = egui::epaint::WHITE_UV;
                        let mut gm = egui::Mesh::default();
                        let gv = gm.vertices.len() as u32;
                        gm.vertices.extend([
                            egui::epaint::Vertex { pos: gradient_rect.left_top(),     uv, color: egui::Color32::BLACK },
                            egui::epaint::Vertex { pos: gradient_rect.right_top(),    uv, color: egui::Color32::WHITE },
                            egui::epaint::Vertex { pos: gradient_rect.right_bottom(), uv, color: egui::Color32::WHITE },
                            egui::epaint::Vertex { pos: gradient_rect.left_bottom(),  uv, color: egui::Color32::BLACK },
                        ]);
                        gm.indices.extend_from_slice(&[gv, gv+1, gv+2, gv, gv+2, gv+3]);
                        sp.add(egui::Shape::Mesh(gm));

                        // Which handle to highlight: dragged one, or nearest within reach when hovering
                        let highlight = (*levels_drag).or_else(|| {
                            strip_resp.hover_pos().and_then(|p| {
                                let dists = [(p.x - black_x).abs(), (p.x - gamma_x).abs(), (p.x - white_x).abs()];
                                dists
                                    .iter()
                                    .enumerate()
                                    .min_by(|a, b| a.1.total_cmp(b.1))
                                    .filter(|(_, d)| **d < 14.0)
                                    .map(|(i, _)| i)
                            })
                        });

                        // Triangle handles (pointing up, sitting below the gradient strip)
                        let ty = gradient_rect.bottom();
                        let by = strip_area.bottom();
                        let mk = |cx: f32, fill: egui::Color32, hot: bool| {
                            let cx = cx.clamp(strip_area.left() + 6.0, strip_area.right() - 6.0);
                            let stroke = if hot {
                                egui::Stroke::new(1.5_f32, egui::Color32::from_gray(220))
                            } else {
                                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(110))
                            };
                            egui::Shape::convex_polygon(
                                vec![egui::pos2(cx, ty), egui::pos2(cx - 6.0, by), egui::pos2(cx + 6.0, by)],
                                fill,
                                stroke,
                            )
                        };
                        sp.add(mk(black_x, egui::Color32::from_gray(20),  highlight == Some(0)));
                        sp.add(mk(gamma_x, egui::Color32::from_gray(128), highlight == Some(1)));
                        sp.add(mk(white_x, egui::Color32::WHITE,          highlight == Some(2)));

                        // Numeric value boxes: black | gamma | white (single compact row)
                        ui.add_space(2.0);
                        ui.columns(3, |cols| {
                            let mut changed = false;
                            cols[0].with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                                changed |= ui.add(
                                    egui::DragValue::new(&mut processor.levels_black)
                                        .range(0.0..=254.0).speed(1.0).max_decimals(0),
                                ).changed();
                            });
                            cols[1].with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                changed |= ui.add(
                                    egui::DragValue::new(&mut processor.levels_gamma)
                                        .range(0.1..=5.0).speed(0.01).fixed_decimals(2),
                                ).changed();
                            });
                            cols[2].with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
                                changed |= ui.add(
                                    egui::DragValue::new(&mut processor.levels_white)
                                        .range(1.0..=255.0).speed(1.0).max_decimals(0),
                                ).changed();
                            });
                            if changed {
                                processor.levels_black = processor.levels_black.min(processor.levels_white - 1.0);
                                *needs_process = true;
                            }
                        });

                        ui.add_space(6.0);
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Open Image…").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("Images", &imgload::all_exts())
                                    .pick_file()
                                {
                                    ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_OPEN_PATH), path));
                                }
                            }
                            if ui
                                .add_enabled(processor.has_image(), egui::Button::new("Auto"))
                                .on_hover_text("Auto: white balance, levels, contrast, shadows/highlights, vibrance")
                                .clicked()
                            {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_AUTO), true));
                            }
                            if ui
                                .add_enabled(
                                    processor.has_image() && current_is_raw,
                                    egui::Button::new("Re-develop RAW"),
                                )
                                .on_hover_text(
                                    "RAWs are auto-developed on open; re-run the universal 16-bit development",
                                )
                                .clicked()
                            {
                                ctx.data_mut(|d| {
                                    d.insert_temp(egui::Id::new(KEY_DEVELOP_RAW), true)
                                });
                            }
                        });
                        if processor.ai_lut_enabled
                            && ui
                                .add(
                                    egui::Slider::new(&mut processor.ai_lut_strength, 0.0..=1.0)
                                        .text("AI strength")
                                        .fixed_decimals(2),
                                )
                                .on_hover_text("How much of the AI adjustment to keep. Lower for a softer, less contrasty result.")
                                .changed()
                        {
                            *needs_process = true;
                        }
                        ui.horizontal_wrapped(|ui| {
                            if ui
                                .add_enabled(processor.has_image(), egui::Button::new("Capture Look"))
                                .on_hover_text("Measure the tone and color of this photo as it looks right now")
                                .clicked()
                            {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_CAPTURE_LOOK), true));
                            }
                            if ui
                                .add_enabled(
                                    processor.has_image() && look_available,
                                    egui::Button::new("Apply Look"),
                                )
                                .on_hover_text("Apply the captured reference's grade with the constrained look model")
                                .clicked()
                            {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_APPLY_LOOK), true));
                            }
                            if ui
                                .add_enabled(
                                    processor.has_image() && look_available,
                                    egui::Button::new(format!("Teach Look Model ({})", app.look_examples.len())),
                                )
                                .on_hover_text(
                                    "Save this approved result and retrain the look model",
                                )
                                .clicked()
                            {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_TEACH_LOOK_MODEL), true));
                            }
                        });
                        if !processor.look.is_empty()
                            && ui
                                .add(
                                    egui::Slider::new(&mut processor.look_strength, 0.0..=1.0)
                                        .text("Look strength")
                                        .fixed_decimals(2),
                                )
                                .on_hover_text("How much of the predicted look to apply")
                                .changed()
                        {
                            *needs_process = true;
                        }
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            let path = current_path.clone();
                            if let Some(path) = path {
                                ui.label(egui::RichText::new("Rating").small().color(egui::Color32::from_gray(140)));
                                for rating in 1..=5u8 {
                                    let filled = meta_map.get(&path).is_some_and(|m| m.rating >= rating);
                                    let label = if filled { "★" } else { "☆" };
                                    if ui.selectable_label(filled, label).clicked() {
                                        cull_actions.push((path.clone(), cull::CullAction::Rating(rating)));
                                    }
                                }
                                if ui.button("0").clicked() {
                                    cull_actions.push((path.clone(), cull::CullAction::Rating(0)));
                                }
                                ui.separator();
                                let pick = meta_map.get(&path).is_some_and(|m| m.flag == cull::Flag::Pick);
                                if ui.selectable_label(pick, "P").clicked() {
                                    cull_actions.push((path.clone(), cull::CullAction::TogglePick));
                                }
                                let reject = meta_map.get(&path).is_some_and(|m| m.flag == cull::Flag::Reject);
                                if ui.selectable_label(reject, "X").clicked() {
                                    cull_actions.push((path.clone(), cull::CullAction::ToggleReject));
                                }
                                ui.separator();
                                for label in [cull::Label::Red, cull::Label::Yellow, cull::Label::Green, cull::Label::Blue] {
                                    let c = label_color(label).unwrap();
                                    let dot = egui::RichText::new("●").color(c);
                                    let active = meta_map.get(&path).is_some_and(|m| m.label == label);
                                    if ui.selectable_label(active, dot).clicked() {
                                        cull_actions.push((path.clone(), cull::CullAction::ToggleLabel(label)));
                                    }
                                }
                            }
                        });
                        ui.separator();
                        if let Some(path) = current_path.clone() {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("TAGS").small().color(egui::Color32::from_gray(140)));
                                let resp = ui.add(
                                    egui::TextEdit::singleline(tag_edit)
                                        .hint_text("comma, separated, tags")
                                        .desired_width(ui.available_width()),
                                );
                                if resp.changed() {
                                    let parsed: Vec<String> = tag_edit
                                        .split(',')
                                        .map(str::trim)
                                        .filter(|s| !s.is_empty())
                                        .map(str::to_string)
                                        .collect();
                                    if let Some(db) = db {
                                        tags::save_tags(db, &path, &parsed);
                                    }
                                    if parsed.is_empty() {
                                        tags_map.remove(&path);
                                    } else {
                                        tags_map.insert(path.clone(), parsed);
                                    }
                                }
                            });
                            ui.separator();
                        }
                        ui.spacing_mut().item_spacing.y = 3.0;

                        // Compact one-line rows: fixed-width label left, slider right
                        macro_rules! slider_row {
                            ($label:expr, $field:expr, $range:expr, $default:expr, $integer:expr) => {{
                                ui.horizontal(|ui| {
                                    // Fixed-size label slot painted directly, so every
                                    // slider starts and ends at the same x.
                                    let (lrect, _) = ui.allocate_exact_size(
                                        egui::vec2(74.0, 18.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().text(
                                        lrect.left_center(),
                                        egui::Align2::LEFT_CENTER,
                                        $label,
                                        egui::FontId::proportional(10.0),
                                        egui::Color32::from_gray(140),
                                    );
                                    ui.spacing_mut().interact_size.x = 46.0;
                                    ui.spacing_mut().slider_width =
                                        ui.available_width() - 46.0 - ui.spacing().item_spacing.x * 2.0;
                                    let mut s = egui::Slider::new(&mut $field, $range).show_value(true);
                                    if $integer { s = s.integer(); }
                                    let r = ui.add(s);
                                    // r.double_clicked() only fires on the text input (click sense).
                                    // Also check raw input so double-clicking the track/thumb resets too.
                                    let double_clicked = r.double_clicked()
                                        || ctx.input(|i| {
                                            i.pointer.button_double_clicked(egui::PointerButton::Primary)
                                                && i.pointer
                                                    .interact_pos()
                                                    .map(|p| r.rect.contains(p))
                                                    .unwrap_or(false)
                                        });
                                    if double_clicked {
                                        $field = $default;
                                        *needs_process = true;
                                    } else if r.changed() {
                                        *needs_process = true;
                                    }
                                });
                            }};
                        }

                        slider_row!("TEMP",       processor.wb_temp,    -100.0..=100.0, 0.0, true);
                        slider_row!("TINT",       processor.wb_tint,    -100.0..=100.0, 0.0, true);
                        ui.separator();
                        slider_row!("EXPOSURE",   processor.exposure,   -3.0..=3.0,     0.0, false);
                        slider_row!("BRIGHTNESS", processor.brightness, -100.0..=100.0, 0.0, true);
                        slider_row!("CONTRAST",   processor.contrast,   -100.0..=100.0, 0.0, true);
                        slider_row!("SATURATION", processor.saturation, -100.0..=100.0, 0.0, true);
                        slider_row!("VIBRANCE",   processor.vibrance,   -100.0..=100.0, 0.0, true);
                        ui.separator();
                        slider_row!("BLACKS",     processor.blacks,     -100.0..=100.0, 0.0, true);
                        slider_row!("SHADOWS",    processor.shadows,    -100.0..=100.0, 0.0, true);
                        slider_row!("HIGHLIGHTS", processor.highlights, -100.0..=100.0, 0.0, true);
                        slider_row!("WHITES",     processor.whites,     -100.0..=100.0, 0.0, true);
                        ui.separator();
                        slider_row!("BLUR",       processor.blur_radius,         0.0..=15.0, 0.0, true);
                        slider_row!("SHARPEN",    processor.unsharp_strength,    0.0..=3.0,  0.0, false);
                        slider_row!("SHARP RAD",  processor.unsharp_blur_radius, 1.0..=10.0, 2.0, true);
                        ui.separator();
                        slider_row!("VIGNETTE",   processor.vignette,     -100.0..=100.0, 0.0,  true);
                        slider_row!("VIG MID",    processor.vignette_mid, 0.0..=100.0,    50.0, true);
                        ui.separator();

                        ui.label(egui::RichText::new("PRESET").small().color(egui::Color32::from_gray(140)));
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("preset_picker")
                                .selected_text(if preset_name.is_empty() { "Choose…" } else { preset_name.as_str() })
                                .width(150.0)
                                .show_ui(ui, |ui| {
                                    for name in presets.iter() {
                                        if ui.selectable_label(preset_name == name, name).clicked() {
                                            preset_name.clone_from(name);
                                            if let Some(state) = presets_dir.and_then(|d| presets::load(d, name)) {
                                                processor.apply_edit_state(&state);
                                                *needs_process = true;
                                            }
                                        }
                                    }
                                });
                            let can_delete = presets.iter().any(|n| n == preset_name);
                            if ui.add_enabled(can_delete, egui::Button::new("🗑")).clicked() {
                                if let Some(dir) = presets_dir {
                                    let _ = presets::delete(dir, preset_name);
                                    *presets = presets::list(dir);
                                    preset_name.clear();
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(preset_name)
                                    .hint_text("name…")
                                    .desired_width(150.0),
                            );
                            if ui
                                .add_enabled(
                                    processor.has_image() && !preset_name.trim().is_empty(),
                                    egui::Button::new("Save"),
                                )
                                .clicked()
                            {
                                if let Some(dir) = presets_dir {
                                    if presets::save(dir, preset_name.trim(), &processor.edit_state()).is_ok() {
                                        *presets = presets::list(dir);
                                    }
                                }
                            }
                        });

                        ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                            ui.add_space(8.0);
                            if ui.add_enabled(
                                processor.has_image(),
                                egui::Button::new("Export…").min_size(egui::vec2(228.0, 0.0)),
                            ).clicked() {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_SHOW_EXPORT), true));
                            }
                            if let Some((done, total)) = *export_progress {
                                ui.label(
                                    egui::RichText::new(format!("Exporting {done} of {total}…"))
                                        .small()
                                        .color(egui::Color32::from_rgb(150, 190, 240)),
                                );
                            }
                            // bottom_up layout: added after Export, so it sits above it
                            if ui.add_enabled(
                                processor.has_image(),
                                egui::Button::new("Reset All Edits (keeps RAW development)").min_size(egui::vec2(228.0, 0.0)),
                            ).clicked() {
                                processor.apply_edit_state(&EditState::default());
                                if let Some(gpu) = app.gpu.as_ref() {
                                    processor.restore_source(&gpu.queue);
                                }
                                *needs_process = true; // re-process + persist the reset
                            }
                            ui.separator();

                            if processor.has_image() {
                                ui.label(
                                    egui::RichText::new("Pinch or ⌘-scroll: zoom · scroll or drag: pan · Double-click: 100% / fit · Hold/Space: original · Hold R: reference")
                                        .small()
                                        .color(egui::Color32::from_gray(120)),
                                );
                            }
                    });
                            });
                });

                if *view == View::Edit {
                    egui::TopBottomPanel::bottom("filmstrip")
                        .exact_height(92.0)
                        .frame(egui::Frame::none().fill(egui::Color32::from_gray(20)))
                        .show(ctx, |ui| {
                            egui::ScrollArea::horizontal().show(ui, |ui| {
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    for &ti in visible {
                                        let entry = &thumbs[ti];
                                        let (rect, resp) = ui.allocate_exact_size(
                                            egui::vec2(104.0, 80.0),
                                            egui::Sense::click(),
                                        );
                                        let is_cur = current_path.as_deref() == Some(entry.path.as_path());
                                        if is_cur && *strip_scroll {
                                            ui.scroll_to_rect(rect, Some(egui::Align::Center));
                                            *strip_scroll = false;
                                        }
                                        if !ui.is_rect_visible(rect) {
                                            continue;
                                        }
                                        let p = ui.painter_at(rect);
                                        p.rect_filled(rect.shrink(2.0), 3.0, egui::Color32::from_gray(14));
                                        let img_area = rect.shrink(4.0);
                                        paint_thumb_cell(
                                            &p,
                                            img_area,
                                            entry.tex.as_ref(),
                                            meta_map.get(&entry.path).copied().unwrap_or_default(),
                                            14.0,
                                            false,
                                        );
                                        if is_cur {
                                            p.rect_stroke(
                                                rect.shrink(2.0),
                                                3.0,
                                                egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(90, 140, 255)),
                                            );
                                        } else if resp.hovered() {
                                            p.rect_stroke(
                                                rect.shrink(2.0),
                                                3.0,
                                                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(110)),
                                            );
                                        }
                                        if resp.clicked() {
                                            *selected = Some(entry.path.clone());
                                            ctx.data_mut(|d| {
                                                d.insert_temp(
                                                    egui::Id::new(KEY_OPEN_PATH),
                                                    entry.path.clone(),
                                                );
                                            });
                                        }
                                    }
                                });
                            });
                        });
                }

                egui::CentralPanel::default()
                    .frame(egui::Frame::none().fill(egui::Color32::from_gray(30)))
                    .show(ctx, |ui| {
                        let panel_rect = ui.max_rect();
                        let panel_size = panel_rect.size();

                        // Full-panel interaction: captures clicks, double-clicks, hover for zoom
                        let response = ui.interact(
                            panel_rect,
                            ui.id().with("img_area"),
                            egui::Sense::click_and_drag(),
                        );

                        let is_dragging = response.dragged();
                        let now = ctx.input(|i| i.time);

                        // Track how long the mouse button has been held (without dragging).
                        // Reset on drag so panning never accidentally shows the original.
                        if is_dragging {
                            *preview_hold_start = None;
                        } else if response.is_pointer_button_down_on() && preview_hold_start.is_none() {
                            *preview_hold_start = Some(now);
                        } else if !response.is_pointer_button_down_on() {
                            *preview_hold_start = None;
                        }

                        // Only show original after 300 ms — fast double-clicks finish in ~200–400 ms
                        // and won't reach the threshold, so they don't flash the original.
                        let held_long_enough = preview_hold_start
                            .is_some_and(|t| now - t > 0.3);
                        // The retouch tool owns the click, so holding the button to
                        // peek at the original would fight it.
                        let show_original = (ctx.input(|i| i.key_down(egui::Key::Space))
                            || held_long_enough)
                            && !*retouch_active;
                        let show_reference = ctx.input(|i| i.key_down(egui::Key::R))
                            && reference_tex_id.is_some();

                        // `[` and `]` resize the brush (see `brush_size_step`
                        // for the layout handling).
                        if *retouch_active {
                            let step: i32 = ctx.input(|i| {
                                i.events
                                    .iter()
                                    .map(|event| match event {
                                        egui::Event::Key {
                                            key,
                                            physical_key,
                                            pressed: true,
                                            ..
                                        } => brush_size_step(key, physical_key),
                                        _ => 0,
                                    })
                                    .sum()
                            });
                            if step != 0 {
                                let min_dim =
                                    processor.image_size.map_or(2048.0, |(w, h)| w.min(h) as f32);
                                *brush_radius = step_brush(min_dim, *brush_radius, step);
                            }
                        }

                        let tex_id = if show_reference {
                            reference_tex_id
                        } else if show_original {
                            original_tex_id
                        } else {
                            image_tex_id
                        };

                        if let Some(tid) = tex_id {
                            let display_size = if show_reference {
                                reference_size
                            } else {
                                processor.image_size
                            };
                            if let Some((iw, ih)) = display_size {
                                let iw = iw as f32;
                                let ih = ih as f32;
                                // What the panel shows: the crop -- unless the
                                // crop tool is open, or an original/reference
                                // is being held, either of which shows the whole
                                // frame. The image texture is always the whole
                                // frame, so showing a crop is just a uv window
                                // onto it, with no resampling.
                                let whole_frame = *crop_active || show_original || show_reference;
                                let uv = if whole_frame || show_reference {
                                    [0.0, 0.0, 1.0, 1.0]
                                } else {
                                    let px = processor::crop_px(processor.crop, (iw as u32, ih as u32));
                                    processor::crop_fractions(px, (iw as u32, ih as u32))
                                };
                                let (vw, vh) = (uv[2] * iw, uv[3] * ih);
                                let fit_scale = (panel_size.x / vw).min(panel_size.y / vh);

                                let (img_offset, img_scale) = if *zoom_fit {
                                    let fw = vw * fit_scale;
                                    let fh = vh * fit_scale;
                                    (egui::vec2(
                                        (panel_size.x - fw) / 2.0,
                                        (panel_size.y - fh) / 2.0,
                                    ), fit_scale)
                                } else {
                                    (*zoom_offset, *zoom_scale)
                                };

                                // Pan: a one-finger drag, or a two-finger trackpad
                                // swipe. egui reports the swipe in "how the
                                // content should move" units -- the same as a
                                // drag delta -- so both just add to the offset
                                // and the image follows the fingers. The crop
                                // tool owns the pointer, so nothing moves then.
                                let delta = if *crop_active {
                                    egui::Vec2::ZERO
                                } else {
                                    pan_delta(
                                        *zoom_fit,
                                        is_dragging.then(|| response.drag_delta()),
                                        if response.hovered() {
                                            ctx.input(|i| i.smooth_scroll_delta)
                                        } else {
                                            egui::Vec2::ZERO
                                        },
                                    )
                                };
                                *zoom_offset += delta;

                                let img_rect = egui::Rect::from_min_size(
                                    panel_rect.min + img_offset,
                                    egui::vec2(vw * img_scale, vh * img_scale),
                                );

                                ui.painter()
                                    .with_clip_rect(panel_rect)
                                    .image(
                                        tid,
                                        img_rect,
                                        egui::Rect::from_min_max(
                                            egui::pos2(uv[0], uv[1]),
                                            egui::pos2(uv[0] + uv[2], uv[1] + uv[3]),
                                        ),
                                        egui::Color32::WHITE,
                                    );

                                let view = ImageView {
                                    rect: img_rect,
                                    uv,
                                    input: (iw, ih),
                                };

                                // Double-click: toggle fit ↔ 100%
                                if response.double_clicked() && !*retouch_active && !*crop_active {
                                    if *zoom_fit {
                                        let cursor = response.hover_pos()
                                            .unwrap_or(panel_rect.center());
                                        let c = cursor - panel_rect.min;
                                        // Image pixel under cursor
                                        let img_px = (c - img_offset) / img_scale;
                                        // At 100%, top-left = cursor - img_px * 1.0
                                        *zoom_offset = c - img_px;
                                        *zoom_scale = 1.0;
                                        *zoom_fit = false;
                                    } else {
                                        *zoom_fit = true;
                                    }
                                }

                                // Crop tool: the rectangle, its handles, and the
                                // drag that resizes it.
                                if *crop_active && !show_original && !show_reference {
                                    let input_px = (iw.max(1.0), ih.max(1.0));
                                    let crop_px = {
                                        let (x, y, w, h) = processor::crop_px(
                                            processor.crop,
                                            (iw as u32, ih as u32),
                                        );
                                        (x as f32, y as f32, (x + w) as f32, (y + h) as f32)
                                    };
                                    let ratio = aspect_ratio(*crop_aspect, input_px);
                                    if let Some(moved) = paint_crop_tool(
                                        ui,
                                        panel_rect,
                                        &view,
                                        crop_px,
                                        input_px,
                                        ratio,
                                        crop_drag,
                                        &response,
                                        ctx,
                                    ) {
                                        let snapped = (
                                            moved.0.round(),
                                            moved.1.round(),
                                            moved.2.round(),
                                            moved.3.round(),
                                        );
                                        processor.crop = processor::crop_from_px(
                                            snapped,
                                            (iw as u32, ih as u32),
                                        );
                                        *needs_process = true;
                                    }
                                }

                                // Retouch layer: draw the dab markers and the brush
                                // cursor, and turn clicks into dabs.
                                if *retouch_active && !show_original && !show_reference {
                                    paint_retouch(
                                        ui,
                                        panel_rect,
                                        &view,
                                        *brush_radius,
                                        &response,
                                        ctx,
                                        &processor.retouch_spots,
                                    );
                                }

                                // Zoom about the cursor. A trackpad pinch arrives
                                // as a Zoom event, which egui reports as a
                                // per-frame factor; ctrl/cmd-scroll takes the
                                // same path (egui routes it here rather than to
                                // the scroll delta, so plain scrolling is free
                                // to pan).
                                if response.hovered() {
                                    let factor = ctx.input(|i| i.zoom_delta());
                                    if (factor - 1.0).abs() > 1e-3 {
                                        let cursor = response.hover_pos()
                                            .unwrap_or(panel_rect.center());
                                        let c = cursor - panel_rect.min;
                                        // Clamp minimum to fit_scale so zooming out never goes smaller than fit
                                        let new_scale =
                                            (img_scale * factor).clamp(fit_scale, 20.0);
                                        *zoom_offset =
                                            zoom_about(c, img_scale, new_scale, img_offset);
                                        *zoom_scale = new_scale;
                                        // Snap to fit mode when at or very near the fit scale
                                        *zoom_fit = new_scale <= fit_scale * 1.03;
                                    }
                                }
                            }
                        } else {
                            ui.painter().text(
                                panel_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                "Drop an image here or use Open Image…",
                                egui::FontId::proportional(14.0),
                                egui::Color32::from_gray(140),
                            );
                        }
                    });
}

#[cfg(test)]
mod tests {
    use super::{
        brush_radius_px, brush_size_step, crop_handle, crop_inside, crop_with_aspect, drag_crop,
        pan_delta, step_brush, take_requests, zoom_about, ImageView, KEY_REDO, KEY_UNDO,
        MIN_CROP_PX,
    };

    // The key handler sets a temp flag and the frame's requests pick it up;
    // this pins that the two names match.
    #[test]
    fn undo_and_redo_requests_are_picked_up() {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_UNDO), true));
            ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_REDO), true));
        });
        let requests = take_requests(&ctx);
        assert!(requests.undo);
        assert!(requests.redo);
        // And an ordinary frame asks for neither.
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let quiet = take_requests(&ctx);
        assert!(!quiet.undo && !quiet.redo);
    }

    // Cropping is all whole-pixel arithmetic in image space; these pin the
    // rules the panel depends on: nothing escapes the image, nothing collapses,
    // and a locked ratio really is locked.
    #[test]
    fn a_crop_drag_stays_inside_the_image_and_keeps_its_minimum() {
        let input = (1000.0, 800.0);
        let rect = (100.0, 100.0, 500.0, 400.0);
        // Dragging the left edge way past the right one must not invert it.
        let dragged = drag_crop(rect, input, 1, egui::vec2(900.0, 0.0), None);
        assert!(dragged.2 - dragged.0 >= MIN_CROP_PX, "{dragged:?}");
        // ...nor may it leave the image.
        assert!(dragged.0 >= 0.0 && dragged.2 <= input.0);
        // Moving slides the rectangle, and the clamp brings it back.
        let moved = drag_crop(rect, input, 0, egui::vec2(-400.0, -400.0), None);
        assert!(moved.0 >= 0.0 && moved.1 >= 0.0);
        assert!((moved.2 - moved.0) - 400.0 < 1e-3);
        assert!((moved.3 - moved.1) - 300.0 < 1e-3);
        // A rect bigger than the image is cut down to it.
        let huge = crop_inside((-500.0, -500.0, 5000.0, 5000.0), input);
        assert_eq!((huge.0, huge.1, huge.2, huge.3), (0.0, 0.0, 1000.0, 800.0));
    }

    #[test]
    fn a_locked_ratio_is_kept_while_dragging() {
        let input = (1000.0, 800.0);
        let rect = (100.0, 100.0, 500.0, 400.0);
        for sides in [1u8, 2, 4, 8, 1 | 4, 2 | 8] {
            let moved = drag_crop(rect, input, sides, egui::vec2(37.0, -21.0), Some(1.0));
            let (w, h) = (moved.2 - moved.0, moved.3 - moved.1);
            assert!(
                (w - h).abs() < 0.5,
                "sides {sides} left a {w}x{h} rect, not square"
            );
        }
        // Locking a ratio from the presets reshapes about the centre.
        let squared = crop_with_aspect(rect, input, 2.0);
        let (w, h) = (squared.2 - squared.0, squared.3 - squared.1);
        assert!((w / h - 2.0).abs() < 1e-3, "{w}x{h}");
        assert!(((squared.0 + squared.2) * 0.5 - 300.0).abs() < 1e-3);
    }

    #[test]
    fn crop_handles_pick_the_edge_the_pointer_is_on() {
        let rect = egui::Rect::from_min_size(egui::pos2(100.0, 100.0), egui::vec2(200.0, 150.0));
        let reach = 10.0;
        assert_eq!(crop_handle(rect, egui::pos2(100.0, 175.0), reach), Some(1)); // left
        assert_eq!(crop_handle(rect, egui::pos2(300.0, 175.0), reach), Some(2)); // right
        assert_eq!(crop_handle(rect, egui::pos2(200.0, 100.0), reach), Some(4)); // top
        assert_eq!(crop_handle(rect, egui::pos2(200.0, 250.0), reach), Some(8)); // bottom
        assert_eq!(crop_handle(rect, egui::pos2(100.0, 100.0), reach), Some(1 | 4)); // corner
        assert_eq!(crop_handle(rect, egui::pos2(200.0, 175.0), reach), Some(0)); // move
        assert_eq!(crop_handle(rect, egui::pos2(500.0, 500.0), reach), None); // elsewhere
    }

    // Markers are stored against the whole image, so with a crop on they have
    // to be placed through it -- and clicking must invert that exactly.
    #[test]
    fn the_image_view_maps_through_a_crop_both_ways() {
        let view = ImageView {
            rect: egui::Rect::from_min_size(egui::pos2(50.0, 20.0), egui::vec2(400.0, 300.0)),
            uv: [0.25, 0.5, 0.5, 0.25],
            input: (4000.0, 2000.0),
        };
        // The crop's own corner is the view's corner.
        let corner = view.to_screen(0.25, 0.5);
        assert!((corner - egui::pos2(50.0, 20.0)).length() < 1e-3, "{corner:?}");
        // And the round trip returns where it started.
        for p in [
            egui::pos2(50.0, 20.0),
            egui::pos2(250.0, 170.0),
            egui::pos2(450.0, 320.0),
        ] {
            let [x, y] = view.to_input(p);
            assert!((view.to_screen(x, y) - p).length() < 1e-2, "{p:?}");
        }
        // One screen pixel is one input pixel times the display scale.
        assert!((view.px_scale() - 400.0 / (0.5 * 4000.0)).abs() < 1e-4);
    }

    // A two-finger swipe and a drag both pan the image in the direction of the
    // gesture (egui's scroll delta already means "how the content should
    // move"), and neither does anything while the image is fitted, where there
    // is nothing to pan to.
    #[test]
    fn a_scroll_or_drag_pans_only_when_zoomed_in() {
        let scroll = egui::vec2(12.0, -30.0);
        assert_eq!(pan_delta(false, None, scroll), scroll);
        assert_eq!(pan_delta(true, None, scroll), egui::Vec2::ZERO);
        let drag = egui::vec2(-5.0, 7.0);
        assert_eq!(pan_delta(false, Some(drag), scroll), drag);
        assert_eq!(pan_delta(true, Some(drag), scroll), egui::Vec2::ZERO);
    }

    // Zooming about the cursor must leave the image point under the cursor
    // where it is -- that is the whole reason the anchor is passed in.
    #[test]
    fn zooming_about_a_point_keeps_that_point_still() {
        let offset = egui::vec2(30.0, -12.0);
        let anchor = egui::vec2(240.0, 180.0);
        let image_point_under_anchor = (anchor - offset) / 2.0;
        for new_scale in [0.5_f32, 1.0, 1.7, 4.0, 20.0] {
            let moved = zoom_about(anchor, 2.0, new_scale, offset);
            let under_anchor = (anchor - moved) / new_scale;
            assert!(
                (under_anchor - image_point_under_anchor).length() < 1e-3,
                "at scale {new_scale} the point under the cursor moved to {under_anchor:?}"
            );
        }
    }

    #[test]
    fn the_brush_steps_and_clamps_to_what_the_search_can_use() {
        // Multiplicative, so the feel is the same at every size.
        assert!((step_brush(1000.0, 40.0, -1) - 32.0).abs() < 1e-4);
        assert!((step_brush(1000.0, 40.0, 1) - 50.0).abs() < 1e-4);
        assert!((step_brush(1000.0, 40.0, 2) - 62.5).abs() < 1e-4);
        // Never below two pixels...
        assert!((step_brush(1000.0, 2.0, -1) - 2.0).abs() < 1e-4);
        // ...and never past a quarter of the shorter side, where the dab search
        // would have no room left to find a source.
        assert!((brush_radius_px(100.0, 1000.0) - 25.0).abs() < 1e-4);
        assert!((step_brush(100.0, 24.0, 1) - 25.0).abs() < 1e-4);
        assert!((step_brush(100.0, 25.0, 5) - 25.0).abs() < 1e-4);
    }

    #[test]
    fn bracket_keys_resize_the_brush_on_any_layout() {
        assert_eq!(brush_size_step(&egui::Key::OpenBracket, &None), -1);
        assert_eq!(brush_size_step(&egui::Key::CloseBracket, &None), 1);
        // Czech layout: the physical bracket keys type `ú` and `)`, so the
        // logical key is not a bracket at all and only the physical one matches.
        assert_eq!(
            brush_size_step(&egui::Key::A, &Some(egui::Key::OpenBracket)),
            -1
        );
        assert_eq!(
            brush_size_step(&egui::Key::A, &Some(egui::Key::CloseBracket)),
            1
        );
        // Unrelated keys, and a bracket press on some other physical key, do nothing.
        assert_eq!(brush_size_step(&egui::Key::A, &None), 0);
        assert_eq!(
            brush_size_step(&egui::Key::OpenBracket, &Some(egui::Key::A)),
            -1
        );
    }
}

// Drives one headless egui frame with the retouch layer over a fixed image
// rectangle and returns the requests it produced. The image rect is placed
// inside the panel so a click maps to a known image pixel.
#[cfg(test)]
fn retouch_frame(
    ctx: &egui::Context,
    events: Vec<egui::Event>,
    spots: &[crate::processor::RetouchSpot],
) -> (UiRequests, usize) {
    let img_rect = egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(400.0, 300.0));
    let input = egui::RawInput {
        events,
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, 600.0),
        )),
        ..Default::default()
    };
    let output = ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let response = ui.interact(
                ui.max_rect(),
                ui.id().with("img_area"),
                egui::Sense::click_and_drag(),
            );
            paint_retouch(
                ui,
                ui.max_rect(),
                &ImageView {
                    rect: img_rect,
                    uv: [0.0, 0.0, 1.0, 1.0],
                    input: (400.0, 300.0),
                },
                40.0,
                &response,
                ctx,
                spots,
            );
        });
    });
    // Every marker is a stroked circle, so counting them pins "each dab is
    // marked" without needing to inspect pixels.
    let circles = output
        .shapes
        .iter()
        .map(|clipped| count_circles(&clipped.shape))
        .sum();
    (take_requests(ctx), circles)
}

#[cfg(test)]
fn count_circles(shape: &egui::Shape) -> usize {
    match shape {
        egui::Shape::Circle(_) => 1,
        egui::Shape::Vec(shapes) => shapes.iter().map(count_circles).sum(),
        _ => 0,
    }
}

#[cfg(test)]
mod pointer_tests {
    use super::retouch_frame;
    use crate::processor::RetouchSpot;

    fn button(pos: egui::Pos2, button: egui::PointerButton, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button,
            pressed,
            modifiers: Default::default(),
        }
    }

    #[test]
    fn clicking_the_photo_queues_a_dab_where_the_pointer_is() {
        let ctx = egui::Context::default();
        // Hover, press, release: egui only reports a click once the button
        // comes back up on the same widget.
        retouch_frame(&ctx, vec![egui::Event::PointerMoved(egui::pos2(300.0, 200.0))], &[]);
        retouch_frame(
            &ctx,
            vec![egui::Event::PointerMoved(egui::pos2(300.0, 200.0)), button(egui::pos2(300.0, 200.0), egui::PointerButton::Primary, true)],
            &[],
        );
        let (requests, _) = retouch_frame(
            &ctx,
            vec![button(egui::pos2(300.0, 200.0), egui::PointerButton::Primary, false)],
            &[],
        );
        // Image rect is 400x300 at (100, 50): (300, 200) is its exact centre,
        // and the 40 px brush is 40 / 300 of the shorter side.
        let dab = requests.retouch.expect("a dab should have been queued");
        assert!((dab[0] - 0.5).abs() < 1e-4, "x = {}", dab[0]);
        assert!((dab[1] - 0.5).abs() < 1e-4, "y = {}", dab[1]);
        assert!((dab[2] - 40.0 / 300.0).abs() < 1e-6, "radius = {}", dab[2]);
    }

    #[test]
    fn right_clicking_a_dab_queues_its_removal() {
        let ctx = egui::Context::default();
        let spots = vec![RetouchSpot {
            x: 0.5,
            y: 0.5,
            radius: 0.1,
            source: [0.2, 0.0],
            gain: [1.0; 3],
            bias: [0.0; 3],
        }];
        // Centre of the first dab: (100, 50) + (0.5, 0.5) * (400, 300).
        let at = egui::pos2(300.0, 200.0);
        retouch_frame(&ctx, vec![egui::Event::PointerMoved(at)], &spots);
        retouch_frame(&ctx, vec![egui::Event::PointerMoved(at), button(at, egui::PointerButton::Secondary, true)], &spots);
        let (requests, _) =
            retouch_frame(&ctx, vec![button(at, egui::PointerButton::Secondary, false)], &spots);
        assert_eq!(requests.retouch_remove, Some(0));
        assert!(requests.retouch.is_none(), "a right click must not place a dab");
    }

    #[test]
    fn a_click_outside_the_photo_does_nothing() {
        let ctx = egui::Context::default();
        let far = egui::pos2(700.0, 550.0);
        retouch_frame(&ctx, vec![egui::Event::PointerMoved(far)], &[]);
        retouch_frame(&ctx, vec![egui::Event::PointerMoved(far), button(far, egui::PointerButton::Primary, true)], &[]);
        let (requests, _) =
            retouch_frame(&ctx, vec![button(far, egui::PointerButton::Primary, false)], &[]);
        assert!(requests.retouch.is_none());
    }
}

#[cfg(test)]
mod marker_tests {
    use super::retouch_frame;
    use crate::processor::RetouchSpot;

    fn dab(x: f32, y: f32, radius: f32) -> RetouchSpot {
        RetouchSpot {
            x,
            y,
            radius,
            source: [0.1, 0.0],
            gain: [1.0; 3],
            bias: [0.0; 3],
        }
    }

    // The dabs are marked while the tool is armed: one stroked ring plus its
    // dark companion per dab, so the count grows with the number of dabs. The
    // call site guards this on the tool being armed, so an unarmed tool paints
    // nothing at all -- and the dabs themselves stay applied either way.
    #[test]
    fn every_dab_is_marked() {
        let ctx = egui::Context::default();
        let at = egui::pos2(300.0, 200.0);
        let move_to = vec![egui::Event::PointerMoved(at)];
        // Warm-up frame: egui only reports hover from the frame after the
        // pointer arrives, and the brush cursor depends on it.
        retouch_frame(&ctx, move_to.clone(), &[]);
        let (_, none) = retouch_frame(&ctx, move_to.clone(), &[]);
        let (_, one) = retouch_frame(&ctx, move_to.clone(), &[dab(0.5, 0.5, 0.1)]);
        let (_, three) = retouch_frame(
            &ctx,
            move_to,
            &[dab(0.5, 0.5, 0.1), dab(0.3, 0.4, 0.1), dab(0.7, 0.6, 0.1)],
        );
        let per_dab = one - none;
        assert!(per_dab > 0, "a dab should be marked on the image");
        assert_eq!(
            three - one,
            2 * per_dab,
            "every dab should add the same number of marks"
        );
    }

    // With dabs overlapping, the one a right-click deletes must be the one
    // drawn on top -- which is the one the markers highlight.
    #[test]
    fn right_click_deletes_the_topmost_dab() {
        let ctx = egui::Context::default();
        let spots = vec![dab(0.5, 0.5, 0.15), dab(0.5, 0.5, 0.15)];
        let at = egui::pos2(300.0, 200.0);
        let button = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Secondary,
            pressed,
            modifiers: Default::default(),
        };
        retouch_frame(&ctx, vec![egui::Event::PointerMoved(at)], &spots);
        retouch_frame(&ctx, vec![egui::Event::PointerMoved(at), button(true)], &spots);
        let (requests, _) = retouch_frame(&ctx, vec![button(false)], &spots);
        assert_eq!(requests.retouch_remove, Some(1));
    }
}
