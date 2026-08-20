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
                                .on_hover_text("Auto levels + brightness/shadows/highlights")
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
                                egui::Button::new("Export PNG…").min_size(egui::vec2(228.0, 0.0)),
                            ).clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("PNG", &["png"])
                                    .set_file_name("output.png")
                                    .save_file()
                                {
                                    ctx.data_mut(|d| d.insert_temp(egui::Id::new(KEY_EXPORT_PATH), path));
                                }
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
                                    egui::RichText::new("Scroll: zoom · Double-click: 100% / fit · Hold/Space: original · Hold R: reference")
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
                        let show_original = ctx.input(|i| i.key_down(egui::Key::Space))
                            || held_long_enough;
                        let show_reference = ctx.input(|i| i.key_down(egui::Key::R))
                            && reference_tex_id.is_some();
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
                                let fit_scale = (panel_size.x / iw).min(panel_size.y / ih);

                                let (img_offset, img_scale) = if *zoom_fit {
                                    let fw = iw * fit_scale;
                                    let fh = ih * fit_scale;
                                    (egui::vec2(
                                        (panel_size.x - fw) / 2.0,
                                        (panel_size.y - fh) / 2.0,
                                    ), fit_scale)
                                } else {
                                    (*zoom_offset, *zoom_scale)
                                };

                                // Pan when zoomed — drag translates the image offset
                                if is_dragging && !*zoom_fit {
                                    *zoom_offset += response.drag_delta();
                                }

                                let img_rect = egui::Rect::from_min_size(
                                    panel_rect.min + img_offset,
                                    egui::vec2(iw * img_scale, ih * img_scale),
                                );

                                ui.painter()
                                    .with_clip_rect(panel_rect)
                                    .image(
                                        tid,
                                        img_rect,
                                        egui::Rect::from_min_max(
                                            egui::pos2(0.0, 0.0),
                                            egui::pos2(1.0, 1.0),
                                        ),
                                        egui::Color32::WHITE,
                                    );

                                // Double-click: toggle fit ↔ 100%
                                if response.double_clicked() {
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

                                // Scroll: zoom at cursor
                                let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                                if scroll.abs() > 0.5 && response.hovered() {
                                    let cursor = response.hover_pos()
                                        .unwrap_or(panel_rect.center());
                                    let c = cursor - panel_rect.min;
                                    let factor = (1.0_f32 + scroll * 0.003).clamp(0.8, 1.25);
                                    // Clamp minimum to fit_scale so scrolling out never goes smaller than fit
                                    let new_scale = (img_scale * factor).clamp(fit_scale, 20.0);
                                    let ratio = new_scale / img_scale;
                                    *zoom_offset = c - (c - img_offset) * ratio;
                                    *zoom_scale = new_scale;
                                    // Snap to fit mode when at or very near the fit scale
                                    *zoom_fit = new_scale <= fit_scale * 1.03;
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
