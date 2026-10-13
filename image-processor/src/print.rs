//! Printing, through CUPS.
//!
//! macOS and Linux print through CUPS, whose command line client is `lp`. The
//! app renders the crop to a JPEG at a sensible size for the chosen paper and
//! hands it to `lp` with the options the dialog collected — which is what the
//! system print dialog would end up doing anyway, minus the system UI.
//!
//! Everything that decides *what* to send is pure and tested here. Only the
//! process launches are not, and those are deliberately thin.
//!
//! Two things learned the hard way, both pinned by tests below:
//!
//! * **`lpstat` output is localised, and `LC_ALL` does not change it.** Apple's
//!   CUPS build takes its messages from the system locale, so a parser keyed on
//!   English prose breaks on the very machine this was written for ("tiskárna
//!   Canon_G3030_series je nečinná"). The parsers here are positional instead:
//!   printer names are the *first* token of an `lpstat -a` line, the second of
//!   an `lpstat -p` line, and whatever follows the colon in `lpstat -d`.
//! * **The paper sizes are the printer's, not a fixed list.** The Canon on that
//!   machine offers A4, 4x6, 5x7, 8x10 and friends with no Letter, Legal or
//!   13x18cm; a built-in list would send a media name it does not know. The
//!   list comes from the printer's own options.

use std::path::Path;
use std::process::Command;

/// A paper the chosen printer accepts. `media` is the CUPS option value; the
/// first entry is always the printer's own default.
#[derive(Clone, PartialEq, Debug)]
pub struct Paper {
    pub label: String,
    /// `None` leaves the paper to the printer, which is the safe default: a
    /// media name this printer does not know would fail the job.
    pub media: Option<String>,
    /// Width and height in millimetres, portrait. `None` when the name carried
    /// no size we could read.
    pub mm: Option<(f32, f32)>,
}

impl Paper {
    /// The entry that says "whatever the printer is loaded with".
    pub fn printer_default() -> Self {
        Self {
            label: "Printer default".to_owned(),
            media: None,
            mm: Some((210.0, 297.0)),
        }
    }
}

/// Print quality, as the IPP enumeration CUPS expects.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Quality {
    Draft,
    Normal,
    High,
}

impl Quality {
    pub fn label(self) -> &'static str {
        match self {
            Self::Draft => "Draft",
            Self::Normal => "Normal",
            Self::High => "High",
        }
    }
    /// 3 = draft, 4 = normal, 5 = high (IPP `print-quality`).
    fn cups_value(self) -> u8 {
        match self {
            Self::Draft => 3,
            Self::Normal => 4,
            Self::High => 5,
        }
    }
    pub fn key(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Normal => "normal",
            Self::High => "high",
        }
    }
    pub fn from_key(key: &str) -> Self {
        match key {
            "draft" => Self::Draft,
            "high" => Self::High,
            _ => Self::Normal,
        }
    }
}

/// Which way up the sheet goes. CUPS wants `orientation-requested`, and leaving
/// it out is the safest: the driver then follows the paper and the image.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Orientation {
    Printer,
    Portrait,
    Landscape,
}

impl Orientation {
    pub fn label(self) -> &'static str {
        match self {
            Self::Printer => "Printer default",
            Self::Portrait => "Portrait",
            Self::Landscape => "Landscape",
        }
    }
    fn cups_value(self) -> Option<u8> {
        match self {
            // 3 = portrait, 4 = landscape.
            Self::Printer => None,
            Self::Portrait => Some(3),
            Self::Landscape => Some(4),
        }
    }
    pub fn landscape(self) -> bool {
        self == Self::Landscape
    }
    pub fn key(self) -> &'static str {
        match self {
            Self::Printer => "printer",
            Self::Portrait => "portrait",
            Self::Landscape => "landscape",
        }
    }
    pub fn from_key(key: &str) -> Self {
        match key {
            "portrait" => Self::Portrait,
            "landscape" => Self::Landscape,
            _ => Self::Printer,
        }
    }
}

/// What the dialog collects. Everything here maps onto `lp` options; nothing
/// here is baked into the image except its size.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PrintOptions {
    /// Index into the paper list the dialog is showing.
    pub paper: usize,
    pub orientation: Orientation,
    pub quality: Quality,
    pub copies: u32,
    pub fit_to_page: bool,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            paper: 0,
            orientation: Orientation::Printer,
            quality: Quality::Normal,
            copies: 1,
            fit_to_page: true,
        }
    }
}

/// The longest side, in pixels, the app will render for a page. This is a bound
/// on the spool file, not a claim about the printed resolution: the printer
/// scales what it is given to the paper.
pub const PRINT_DPI: f32 = 600.0;

/// No print will be rendered larger than this, whatever the paper says.
const MAX_PRINT_PX: u32 = 12000;

/// The pixel box a paper provides at `dpi`.
pub fn page_box(paper: &Paper, landscape: bool, dpi: f32) -> (u32, u32) {
    let (w_mm, h_mm) = paper.mm.unwrap_or((210.0, 297.0));
    let (w_mm, h_mm) = if landscape { (h_mm, w_mm) } else { (w_mm, h_mm) };
    let to_px = |mm: f32| {
        ((mm / 25.4 * dpi).round() as u32).clamp(1, MAX_PRINT_PX)
    };
    (to_px(w_mm), to_px(h_mm))
}

/// Scale `(w, h)` down to fit inside `box_wh`, keeping the aspect. Never
/// enlarges: a small image is sent as it is and the printer scales it up.
pub fn fit_inside(w: u32, h: u32, box_wh: (u32, u32)) -> (u32, u32) {
    if w <= box_wh.0 && h <= box_wh.1 {
        return (w, h);
    }
    let scale = f32::min(
        box_wh.0 as f32 / w.max(1) as f32,
        box_wh.1 as f32 / h.max(1) as f32,
    );
    (
        ((w as f32 * scale).round() as u32).max(1),
        ((h as f32 * scale).round() as u32).max(1),
    )
}

/// A media name without its CUPS variant suffix: `A4.Fullbleed` is A4.
///
/// Only known suffixes come off. Splitting on the first dot instead looks
/// reasonable until a real printer's list arrives with `3.5x5` in it, which
/// becomes `3` -- a size, read as a variant of a paper called three.
fn media_base(name: &str) -> &str {
    const SUFFIXES: [&str; 2] = [".Fullbleed", ".Borderless"];
    let trimmed = name.trim();
    for suffix in SUFFIXES {
        if let Some(base) = trimmed.strip_suffix(suffix) {
            return base;
        }
    }
    trimmed
}

/// Millimetres for a media name: a table for the usual names, and otherwise the
/// name itself when it is a plain `WxH` — `4x6` is inches, `215x345mm` is not.
fn media_mm(name: &str) -> Option<(f32, f32)> {
    let base = media_base(name).to_ascii_lowercase();
    let known = match base.as_str() {
        "a3" => Some((297.0, 420.0)),
        "a4" => Some((210.0, 297.0)),
        "a5" => Some((148.0, 210.0)),
        "a6" => Some((105.0, 148.0)),
        "b5" => Some((176.0, 250.0)),
        "letter" | "na_letter" => Some((215.9, 279.4)),
        "legal" | "na_legal" => Some((215.9, 355.6)),
        "tabloid" => Some((279.4, 431.8)),
        _ => None,
    };
    if known.is_some() {
        return known;
    }
    let (numbers, millimetres) = if let Some(rest) = base.strip_suffix("mm") {
        (rest, true)
    } else {
        (base.as_str(), false)
    };
    let (w, h) = numbers.split_once('x')?;
    let (w, h) = (w.trim().parse::<f32>().ok()?, h.trim().parse::<f32>().ok()?);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    // A paper is never a handful of millimetres, so small bare numbers are
    // inches: `4x6` is 10x15cm, and `215x345` is already millimetres.
    if millimetres || w > 30.0 || h > 30.0 {
        Some((w, h))
    } else {
        Some((w * 25.4, h * 25.4))
    }
}

/// A label for a media name, with the borderless variants called out.
fn media_label(name: &str) -> String {
    let borderless = name != media_base(name);
    let base = media_base(name);
    let pretty = size_label(base).unwrap_or_else(|| base.to_owned());
    if borderless {
        format!("{pretty} (borderless)")
    } else {
        pretty
    }
}

/// `4x6` -> `4 × 6 in`, `215x345mm` -> `215 × 345 mm`, `A4` -> `A4`. The unit is
/// taken from the name, not from a conversion, so nothing is silently changed
/// from imperial to metric.
fn size_label(base: &str) -> Option<String> {
    const NAMED: [&str; 8] = ["a3", "a4", "a5", "a6", "b5", "letter", "legal", "tabloid"];
    let base = media_base(base);
    let lower = base.to_ascii_lowercase();
    if NAMED.contains(&lower.as_str()) {
        return Some(base.to_owned());
    }
    let (numbers, named_mm) = match lower.strip_suffix("mm") {
        Some(rest) => (rest.to_owned(), true),
        None => (lower, false),
    };
    let (w, h) = numbers.split_once('x')?;
    let (w, h) = (w.parse::<f32>().ok()?, h.parse::<f32>().ok()?);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    // A paper is never a handful of millimetres, so small bare numbers are
    // inches: `4x6` is 10x15cm while `215x345` is already millimetres.
    let unit = if named_mm || w > 30.0 || h > 30.0 {
        "mm"
    } else {
        "in"
    };
    let trim = |v: f32| {
        if v.fract() == 0.0 {
            format!("{}", v as i64)
        } else {
            format!("{v}")
        }
    };
    Some(format!("{} × {} {}", trim(w), trim(h), unit))
}

/// The papers a printer offers, from `lpoptions -l`.
///
/// The first entry is always the printer's own default (no `media` option at
/// all); the rest are the values its description lists, with the one it starts
/// on marked. Empty input gives just the default, which always works.
pub fn parse_papers(lpoptions: &str) -> Vec<Paper> {
    let mut papers = vec![Paper::printer_default()];
    let line = lpoptions
        .lines()
        .find(|line| {
            let head = line.split(':').next().unwrap_or("");
            head.starts_with("PageSize") || head.starts_with("MediaSize")
        });
    let Some(line) = line else {
        return papers;
    };
    let Some((_, values)) = line.split_once(':') else {
        return papers;
    };
    let mut default = None;
    let mut named = Vec::new();
    for value in values.split_whitespace() {
        let (is_default, name) = match value.strip_prefix('*') {
            Some(name) => (true, name),
            None => (false, value),
        };
        if name.is_empty() || named.iter().any(|p: &Paper| p.media.as_deref() == Some(name)) {
            continue;
        }
        if is_default {
            default = Some(named.len());
        }
        named.push(Paper {
            label: media_label(name),
            media: Some(name.to_owned()),
            mm: media_mm(name),
        });
    }
    if let Some(index) = default {
        papers[0].label = format!("Printer default ({})", named[index].label);
    }
    papers.extend(named);
    papers
}

/// The printers CUPS knows, and which one it would use by default, from the
/// three `lpstat` queries the caller ran.
///
/// Positional on purpose: the wording is localised (and `LC_ALL` does not
/// change that on macOS), but the shape does not. `lpstat -a` puts the name
/// first, `lpstat -p` second, and `-d` names the default after the colon.
pub fn parse_printers(lpstat_a: &str, lpstat_p: &str, lpstat_d: &str) -> (Vec<String>, Option<String>) {
    let mut names: Vec<String> = Vec::new();
    let mut push = |name: &str| {
        let name = name.trim();
        // A queue name never contains whitespace, and CUPS caps it at 127.
        let plausible =
            !name.is_empty() && !name.contains(char::is_whitespace) && name.len() <= 127;
        if plausible && !names.iter().any(|n| n == name) {
            names.push(name.to_owned());
        }
    };
    for line in lpstat_a.lines() {
        // "<name> accepting requests since ..."
        if let Some(name) = line.split_whitespace().next() {
            push(name);
        }
    }
    for line in lpstat_p.lines() {
        // "printer <name> is idle" / "tiskárna <name> je nečinná"
        let mut words = line.split_whitespace();
        words.next();
        if let Some(name) = words.next() {
            push(name);
        }
    }
    // "system default destination: <name>" / "výchozí systémový cíl: <name>"
    let default = lpstat_d
        .split_once(':')
        .map(|(_, name)| name.trim().to_owned())
        .filter(|name| !name.is_empty())
        .filter(|name| names.iter().any(|n| n == name));
    (names, default)
}

/// The argv for `lp`: printer, copies, paper, orientation, quality, scaling,
/// then the file.
pub fn lp_args(printer: Option<&str>, options: &PrintOptions, paper: &Paper, path: &Path) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    if let Some(printer) = printer.filter(|p| !p.is_empty()) {
        args.push("-d".to_owned());
        args.push(printer.to_owned());
    }
    if options.copies > 1 {
        args.push("-n".to_owned());
        args.push(options.copies.to_string());
    }
    let mut option = |value: String| {
        args.push("-o".to_owned());
        args.push(value);
    };
    if let Some(media) = &paper.media {
        option(format!("media={media}"));
    }
    if let Some(value) = options.orientation.cups_value() {
        option(format!("orientation-requested={value}"));
    }
    option(format!("print-quality={}", options.quality.cups_value()));
    if options.fit_to_page {
        option("fit-to-page".to_owned());
    }
    // `--` first, so a path can never be read as an option.
    args.push("--".to_owned());
    args.push(path.to_string_lossy().into_owned());
    args
}

/// The printers this machine has, and its default. Empty on anything that is
/// not CUPS (a caller then offers the system default alone).
pub fn available_printers() -> (Vec<String>, Option<String>) {
    let query = |args: &[&str]| -> String {
        Command::new("lpstat")
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
            .unwrap_or_default()
    };
    parse_printers(&query(&["-a"]), &query(&["-p"]), &query(&["-d"]))
}

/// The papers `printer` offers. With no printer named, CUPS answers for its own
/// default, which is what the dialog starts on.
pub fn papers_for(printer: Option<&str>) -> Vec<Paper> {
    let mut command = Command::new("lpoptions");
    if let Some(printer) = printer.filter(|p| !p.is_empty()) {
        command.arg("-p").arg(printer);
    }
    let listed = command
        .arg("-l")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_default();
    parse_papers(&listed)
}

/// Hand `path` to CUPS. Returns what `lp` said, or its complaint — `lp` reports
/// failures (unknown printer, stopped queue, unreadable file) on stderr, in the
/// user's own language, and that text is worth showing verbatim.
pub fn send(
    printer: Option<&str>,
    options: &PrintOptions,
    paper: &Paper,
    path: &Path,
) -> Result<String, String> {
    let output = Command::new("lp")
        .args(lp_args(printer, options, paper, path))
        .output()
        .map_err(|e| format!("could not run lp: {e}"))?;
    let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).trim().to_owned();
    if output.status.success() {
        let said = text(&output.stdout);
        Ok(if said.is_empty() {
            "sent to the printer".to_owned()
        } else {
            said
        })
    } else {
        let said = text(&output.stderr);
        Err(if said.is_empty() {
            format!("lp exited with {}", output.status)
        } else {
            said
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // The real output of this machine, which localises its messages and ignores
    // LC_ALL. The shape is what the parsing relies on, not the wording.
    const CZ_A: &str = "Canon_G3030_series přijímá požadavky od Mon Oct  5 09:28:57 2026\n\
                        HP_LaserJet_P2055dn přijímá požadavky od Mon Sep 21 13:44:56 2026\n";
    const CZ_P: &str = "tiskárna Canon_G3030_series je nečinná.  zapnuta od Mon Oct  5 09:28:57 2026\n\
                        tiskárna HP_LaserJet_P2055dn je nečinná.  zapnuta od Mon Sep 21 13:44:56 2026\n";
    const CZ_D: &str = "výchozí systémový cíl: Canon_G3030_series\n";
    const EN_A: &str = "Office accepting requests since Mon 21 Sep 2026\n";
    const EN_P: &str = "printer Office is idle.  enabled since Mon 21 Sep 2026\n\
                        printer Old disabled since Mon 01 Jan 2026 -\n";
    const EN_D: &str = "system default destination: Office\n";

    #[test]
    fn printers_parse_in_any_locale() {
        let (names, default) = parse_printers(CZ_A, CZ_P, CZ_D);
        assert_eq!(names, vec!["Canon_G3030_series", "HP_LaserJet_P2055dn"]);
        assert_eq!(default.as_deref(), Some("Canon_G3030_series"));

        // English, and a disabled queue that only `-p` reports.
        let (names, default) = parse_printers(EN_A, EN_P, EN_D);
        assert_eq!(names, vec!["Office", "Old"]);
        assert_eq!(default.as_deref(), Some("Office"));

        // Nothing running: no printers, and no default invented.
        let (names, default) = parse_printers("", "", "");
        assert!(names.is_empty());
        assert_eq!(default, None);
        // A default that is not in the list is not reported either.
        let (_, default) = parse_printers(EN_A, EN_P, "system default destination: Gone\n");
        assert_eq!(default, None);
        // No default at all is not a name.
        let (_, default) = parse_printers(EN_A, EN_P, "no system default destination\n");
        assert_eq!(default, None);
        // Localised "no default" carries no colon either.
        let (_, default) = parse_printers(CZ_A, CZ_P, "žádný výchozí systémový cíl\n");
        assert_eq!(default, None);
    }

    // The Canon's own list, abridged. Note what is *not* there: no Letter, no
    // Legal, no 13x18cm. A built-in list would offer paper it cannot load.
    const CANON_OPTIONS: &str = "PageSize/Media Size: 215x345mm 3.5x5 3.5x5.Fullbleed 4x6 \
        4x6.Fullbleed 5x5 5x5.Fullbleed 5x7 *A4 A4.Fullbleed A5 A6\n";

    #[test]
    fn papers_come_from_the_printer_and_keep_its_default() {
        let papers = parse_papers(CANON_OPTIONS);
        assert_eq!(papers[0].media, None, "the first entry always works");
        assert_eq!(papers[0].label, "Printer default (A4)");
        assert_eq!(papers.len(), 1 + 12);
        // A size with a decimal point is a size, not a variant of a paper
        // called three -- this one shipped until a real printer's list was read.
        let three_and_a_half = papers
            .iter()
            .find(|p| p.media.as_deref() == Some("3.5x5"))
            .unwrap();
        assert_eq!(three_and_a_half.label, "3.5 × 5 in");
        assert_eq!(three_and_a_half.mm, Some((88.9, 127.0)));
        assert!(papers.iter().any(|p| p.media.as_deref() == Some("4x6")));
        let four_by_six = papers
            .iter()
            .find(|p| p.media.as_deref() == Some("4x6"))
            .unwrap();
        assert_eq!(four_by_six.label, "4 × 6 in");
        assert_eq!(four_by_six.mm, Some((101.6, 152.4)));
        let borderless = papers
            .iter()
            .find(|p| p.media.as_deref() == Some("5x5.Fullbleed"))
            .unwrap();
        assert_eq!(borderless.label, "5 × 5 in (borderless)");
        let mm = papers
            .iter()
            .find(|p| p.media.as_deref() == Some("215x345mm"))
            .unwrap();
        assert_eq!(mm.label, "215 × 345 mm");
        // Nothing to read: the default alone, which still prints.
        assert_eq!(parse_papers("").len(), 1);
        assert_eq!(parse_papers("PageSize/Media Size:\n").len(), 1);
        // Duplicates in one list are not offered twice.
        assert_eq!(parse_papers("MediaSize/Size: A4 A4 A5\n").len(), 3);
    }

    #[test]
    fn media_sizes_read_from_the_name() {
        assert_eq!(media_mm("A4"), Some((210.0, 297.0)));
        assert_eq!(media_mm("a4"), Some((210.0, 297.0)));
        assert_eq!(media_mm("A4.Fullbleed"), Some((210.0, 297.0)));
        assert_eq!(media_mm("3.5x5"), Some((88.9, 127.0)));
        assert_eq!(media_mm("5.5x8.5"), Some((139.7, 215.9)));
        assert_eq!(media_mm("3.5x5.Fullbleed"), Some((88.9, 127.0)));
        // Only known suffixes come off, so a dotted size survives intact.
        assert_eq!(media_base("3.5x5"), "3.5x5");
        assert_eq!(media_base("A4.Fullbleed"), "A4");
        assert_eq!(media_base("Custom.WIDTHxHEIGHT"), "Custom.WIDTHxHEIGHT");
        assert_eq!(media_mm("4x6"), Some((101.6, 152.4)));
        assert_eq!(media_mm("215x345mm"), Some((215.0, 345.0)));
        assert_eq!(media_mm("89x89mm"), Some((89.0, 89.0)));
        assert_eq!(media_mm("nonsense"), None);
        assert_eq!(media_mm("0x6"), None);
    }

    #[test]
    fn the_print_box_follows_the_paper_and_the_orientation() {
        let a4 = Paper {
            label: "A4".into(),
            media: Some("A4".into()),
            mm: Some((210.0, 297.0)),
        };
        assert_eq!(page_box(&a4, false, 600.0), (4961, 7016));
        assert_eq!(page_box(&a4, true, 600.0), (7016, 4961));
        // A paper whose size we could not read still yields something sane.
        let unknown = Paper {
            label: "?".into(),
            media: Some("?".into()),
            mm: None,
        };
        assert_eq!(page_box(&unknown, false, 600.0), (4961, 7016));
        // Never enormous, whatever the paper claims.
        let wild = Paper {
            label: "wild".into(),
            media: None,
            mm: Some((100000.0, 100000.0)),
        };
        assert_eq!(page_box(&wild, false, 600.0), (MAX_PRINT_PX, MAX_PRINT_PX));
    }

    #[test]
    fn fitting_a_page_keeps_the_aspect_and_only_shrinks() {
        // A 6000x4000 frame on A4 at 600 dpi: width is the limit.
        assert_eq!(fit_inside(6000, 4000, (4961, 7016)), (4961, 3307));
        // A landscape box is roomier than that, and fitting never enlarges.
        assert_eq!(fit_inside(6000, 4000, (7016, 4961)), (6000, 4000));
        // A tall image in a landscape box: the height is the limit.
        assert_eq!(fit_inside(4000, 9000, (7016, 4961)), (2205, 4961));
        // Smaller than the page: sent as it is, the printer scales it up.
        assert_eq!(fit_inside(1200, 800, (4961, 7016)), (1200, 800));
        assert_eq!(fit_inside(1, 4000, (100, 100)), (1, 100));
    }

    #[test]
    fn the_lp_command_says_what_the_dialog_says() {
        let papers = parse_papers(CANON_OPTIONS);
        let a4 = papers
            .iter()
            .find(|p| p.media.as_deref() == Some("A4"))
            .unwrap()
            .clone();
        let path = PathBuf::from("/Users/someone/.image-processor/print.jpg");

        // Everything at its default: one comment-free, single-copy job with the
        // paper left to the printer.
        let plain = lp_args(None, &PrintOptions::default(), &papers[0], &path);
        assert_eq!(
            plain,
            vec![
                "-o",
                "print-quality=4",
                "-o",
                "fit-to-page",
                "--",
                "/Users/someone/.image-processor/print.jpg"
            ]
        );

        let options = PrintOptions {
            paper: 1,
            orientation: Orientation::Landscape,
            quality: Quality::High,
            copies: 3,
            fit_to_page: false,
        };
        let full = lp_args(Some("Canon_G3030_series"), &options, &a4, &path);
        assert_eq!(
            full,
            vec![
                "-d",
                "Canon_G3030_series",
                "-n",
                "3",
                "-o",
                "media=A4",
                "-o",
                "orientation-requested=4",
                "-o",
                "print-quality=5",
                "--",
                "/Users/someone/.image-processor/print.jpg"
            ]
        );
        // Portrait is orientation 3, and an empty printer name is ignored.
        let portrait = PrintOptions {
            orientation: Orientation::Portrait,
            ..PrintOptions::default()
        };
        let args = lp_args(Some(""), &portrait, &papers[0], &path);
        assert!(args.contains(&"orientation-requested=3".to_owned()));
        assert!(!args.contains(&"-d".to_owned()));
        assert!(!args.contains(&"-n".to_owned()), "one copy needs no -n");
    }

    #[test]
    fn quality_and_orientation_keys_round_trip() {
        for quality in [Quality::Draft, Quality::Normal, Quality::High] {
            assert_eq!(Quality::from_key(quality.key()), quality);
        }
        for orientation in [
            Orientation::Printer,
            Orientation::Portrait,
            Orientation::Landscape,
        ] {
            assert_eq!(Orientation::from_key(orientation.key()), orientation);
        }
        assert_eq!(Quality::from_key("nonsense"), Quality::Normal);
        assert_eq!(Orientation::from_key("nonsense"), Orientation::Printer);
    }
}

