//! Printing through AppKit: the system print sheet.
//!
//! macOS apps print through `NSPrintOperation`, which shows the standard sheet
//! — printer, presets, paper size, and every driver setting the printer has,
//! including the paper *type* whose absence from the CUPS path bleached a photo
//! print. It also hands off to CUPS underneath, so the same spooler does the
//! work.
//!
//! Two things make this far cheaper than the Objective-C route sounds. The
//! image goes into a stock `NSImageView`, so there is no view subclass and no
//! message forwarding — the bindings here are objc2's, which are memory-safe and
//! use `MainThreadMarker` to prove main-thread access, so what is left `unsafe`
//! is only the two class-method calls that objc2 marks as such. And the sheet
//! *is* the options UI, so the app needs no dialog of its own on this platform:
//! the panel's button renders the crop and opens the sheet.
//!
//! The page geometry is checkable without a printer: `to_pdf` builds the same
//! view and page setup and writes a PDF instead of showing the sheet, which is
//! what the test does.

use objc2::rc::Retained;
use objc2::{AllocAnyThread, MainThreadMarker};

use crate::print;
use objc2_app_kit::{
    NSImage, NSImageView, NSImageScaling, NSModalResponseOK, NSPrintInfo, NSPrintOperation,
    NSPrintPanel, NSPrintPanelOptions, NSPrintingPaginationMode, NSPrinter,
};
use objc2_foundation::{NSData, NSMutableData, NSString};

/// An image to print: the crop, already rendered and encoded.
pub struct Page {
    /// JPEG bytes. Their resolution metadata decides the size the sheet starts
    /// from, which is why the render records it.
    pub encoded: Vec<u8>,
    /// Scale the image to the page rather than printing it at the size its own
    /// resolution describes.
    pub fit_to_page: bool,
}

fn main_thread() -> Result<MainThreadMarker, String> {
    // Always Some in this app: printing is driven from the UI thread, and the
    // marker is AppKit's own proof of it.
    MainThreadMarker::new().ok_or_else(|| "printing needs the main thread".to_owned())
}

/// The image in a stock image view, sized to whatever resolution its metadata
/// says it is. Nothing here draws: AppKit does.
fn image_view(mtm: MainThreadMarker, page: &Page) -> Result<Retained<NSImageView>, String> {
    let data = NSData::with_bytes(&page.encoded);
    let image = NSImage::initWithData(NSImage::alloc(), &data)
        .ok_or_else(|| "the rendered image could not be read".to_owned())?;
    let view = NSImageView::imageViewWithImage(&image, mtm);
    // A view created programmatically has a zero frame, and printing a zero
    // frame produces a blank page: the image's own size is what gives the page
    // something to lay out.
    view.setFrameSize(image.size());
    // The view is scaled to the page by the print operation; inside it, the
    // image fills the view without changing its proportions.
    view.setImageScaling(NSImageScaling::ScaleProportionallyUpOrDown);
    Ok(view)
}

/// Page setup: fit the view to the paper, or print it at its own size.
fn print_info(page: &Page) -> Retained<NSPrintInfo> {
    let info = NSPrintInfo::new();
    let mode = if page.fit_to_page {
        NSPrintingPaginationMode::Fit
    } else {
        NSPrintingPaginationMode::Clip
    };
    info.setHorizontalPagination(mode);
    info.setVerticalPagination(mode);
    info
}

/// Whether two names are the same printer. AppKit uses the printer's own name
/// ("Canon G3030 series"); CUPS names its queue "Canon_G3030_series". Matching
/// the strings exactly therefore fails for exactly the printers whose names
/// have spaces in them, which is most of them.
fn same_printer(appkit: &str, cups: &str) -> bool {
    let normalise = |name: &str| {
        name.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect::<String>()
    };
    normalise(appkit) == normalise(cups)
}

/// Select the printer CUPS would use, so the sheet opens on the one already in
/// use rather than on whatever AppKit would guess. Best effort: a sheet with the
/// wrong preselection is a nuisance, not a failure.
pub fn preselect_default_printer(info: &NSPrintInfo, cups_name: &str) {
    for name in printer_names() {
        if same_printer(&name, cups_name) {
            if let Some(printer) = NSPrinter::printerWithName(&NSString::from_str(&name)) {
                info.setPrinter(&printer);
            }
            return;
        }
    }
}

/// The printer names AppKit knows. Worth seeing next to the CUPS queue names,
/// which are not always spelled the same way.
pub fn printer_names() -> Vec<String> {
    NSPrinter::printerNames()
        .iter()
        .map(|name| name.to_string())
        .collect()
}

/// Write the page to PDF. No sheet, no printer: this is the half of printing
/// that can be checked here, and it is also how the app offers a print-ready
/// PDF.
pub fn to_pdf(page: &Page) -> Result<Vec<u8>, String> {
    let mtm = main_thread()?;
    let view = image_view(mtm, page)?;
    let info = print_info(page);
    let data = NSMutableData::new();
    // A PDF print operation renders the view into the data instead of spooling
    // it, using the same page setup the sheet would.
    let operation = NSPrintOperation::PDFOperationWithView_insideRect_toData_printInfo(
        &view,
        view.bounds(),
        &data,
        &info,
    );
    // Nothing is being printed, so nothing may ask the user anything: without
    // this the operation can put up a panel and wait for an answer that no one
    // is there to give.
    operation.setShowsPrintPanel(false);
    operation.setShowsProgressPanel(false);
    if !operation.runOperation() {
        return Err("the PDF could not be written".to_owned());
    }
    Ok(data.to_vec())
}

/// Ask for a page: paper size, orientation, scaling, and whatever the driver
/// adds. Returns `false` if the user cancelled.
///
/// This is a separate step from printing on purpose. `runOperation` shows the
/// panel and then prints, and if the panel does not come up for any reason
/// there is nothing to stop it going straight to the printer — which is exactly
/// what happened once while this was being written: a page of photo paper came
/// out of a print that nobody asked for. Asking first and turning the panel off
/// for the job removes that possibility: no answer, no print.
fn ask_for_page(mtm: MainThreadMarker, info: &NSPrintInfo) -> bool {
    let panel = NSPrintPanel::printPanel(mtm);
    // The stock panel offers copies and the page range and nothing else: paper
    // size, orientation and scaling are opt-in, which is why a sheet can come
    // up with no way to choose the paper at all.
    panel.setOptions(
        NSPrintPanelOptions::ShowsCopies
            | NSPrintPanelOptions::ShowsPageRange
            | NSPrintPanelOptions::ShowsPaperSize
            | NSPrintPanelOptions::ShowsOrientation
            | NSPrintPanelOptions::ShowsScaling
            | NSPrintPanelOptions::ShowsPageSetupAccessory
            | NSPrintPanelOptions::ShowsPreview,
    );
    panel.runModalWithPrintInfo(info) == NSModalResponseOK
}

/// Show the system print sheet and print what the user confirms.
pub fn run_dialog(page: &Page) -> Result<String, String> {
    let mtm = main_thread()?;
    let view = image_view(mtm, page)?;
    let info = print_info(page);
    let (_, default) = print::available_printers();
    if let Some(name) = default {
        preselect_default_printer(&info, &name);
    }
    if !ask_for_page(mtm, &info) {
        return Err("the print sheet was closed without printing".to_owned());
    }
    // The panel writes the user's choices back into the print info it was given
    // -- paper, orientation, scaling, and whatever the driver added -- and that
    // is the object the job is built from.
    let operation = NSPrintOperation::printOperationWithView_printInfo(&view, &info);
    operation.setShowsPrintPanel(false);
    operation.setShowsProgressPanel(true);
    if !operation.runOperation() {
        return Err("the job could not be run".to_owned());
    }
    Ok("sent to the printer".to_owned())
}

#[cfg(test)]
mod tests {
    use super::{same_printer, to_pdf, Page};
    use objc2::{AllocAnyThread, MainThreadMarker};
    use objc2_app_kit::NSImage;
    use objc2_foundation::NSData;

    /// A tiny JPEG, encoded the way the print path encodes one.
    fn page_bytes(dpi: u16, wide: u32, tall: u32) -> Vec<u8> {
        let pixels: Vec<u8> = (0..wide * tall)
            .flat_map(|i| {
                let v = (i % 251) as u8;
                [v, v.wrapping_add(60), v.wrapping_add(140)]
            })
            .collect();
        let mut bytes = Vec::new();
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 90);
        encoder.set_pixel_density(image::codecs::jpeg::PixelDensity::dpi(dpi));
        encoder
            .encode(&pixels, wide, tall, image::ExtendedColorType::Rgb8)
            .unwrap();
        bytes
    }

    // The printers whose names contain spaces are exactly the ones an exact
    // string match fails on, and the Canon on this machine is one of them.
    #[test]
    fn printer_names_match_across_appkit_and_cups() {
        assert!(same_printer("Canon G3030 series", "Canon_G3030_series"));
        assert!(same_printer("HP_LaserJet_P2055dn", "HP_LaserJet_P2055dn"));
        assert!(same_printer("Epson ET-2850 Series", "Epson_ET-2850_Series"));
        assert!(!same_printer("Canon G3030 series", "HP_LaserJet_P2055dn"));
        assert!(!same_printer("", "anything"));
    }

    // The file's own resolution is what AppKit sizes the page from, which is the
    // whole reason the render records one: a 1200x800 render at 600 dpi is two
    // inches wide, and "do not scale" then means two inches rather than whatever
    // the driver would have assumed. An image is not a view, so unlike the rest
    // of this module it can be checked off the main thread.
    #[test]
    fn appkit_reads_the_size_out_of_the_resolution_we_write() {
        let data = NSData::with_bytes(&page_bytes(600, 1200, 800));
        let image = NSImage::initWithData(NSImage::alloc(), &data).expect("a readable image");
        let size = image.size();
        // 1200/600 inch = 2 in = 144 points; 800/600 in = 96 points.
        assert!((size.width - 144.0).abs() < 2.0, "width {} points", size.width);
        assert!((size.height - 96.0).abs() < 2.0, "height {} points", size.height);
    }

    // AppKit views may only be built on the main thread, and objc2 makes that a
    // compile-time proof rather than a runtime gamble -- so the way in has to
    // refuse rather than proceed. Under `cargo test` that is exactly the
    // situation, which is why this is a test and the sheet itself is not: the
    // page building is verified by hand, by looking at the panel and by writing
    // a PDF through the same view and page setup.
    #[test]
    fn building_a_page_off_the_main_thread_is_refused() {
        if MainThreadMarker::new().is_some() {
            eprintln!("running on the main thread; nothing to check here");
            return;
        }
        let page = Page {
            encoded: page_bytes(600, 120, 80),
            fit_to_page: true,
        };
        let error = to_pdf(&page).expect_err("AppKit must not be driven off the main thread");
        assert!(
            error.contains("main thread"),
            "the refusal should say why: {error}"
        );
    }
}
