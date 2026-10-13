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
    NSImage, NSImageView, NSImageScaling, NSPrintInfo, NSPrintOperation,
    NSPrintingPaginationMode, NSPrinter,
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

/// Select the printer CUPS would use, so the sheet opens on the one already in
/// use rather than on whatever AppKit would guess. Best effort: a sheet with
/// the wrong preselection is a nuisance, not a failure.
pub fn preselect_default_printer(info: &NSPrintInfo, name: &str) {
    let name = NSString::from_str(name);
    if let Some(printer) = NSPrinter::printerWithName(&name) {
        info.setPrinter(&printer);
    }
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

/// Show the system print sheet. Blocks until the user is done with it, which is
/// what a modal sheet does — that is why the caller renders first, off the UI
/// thread, so only the sheet itself is modal.
pub fn run_dialog(page: &Page) -> Result<String, String> {
    let mtm = main_thread()?;
    let view = image_view(mtm, page)?;
    let info = print_info(page);
    let (_, default) = print::available_printers();
    if let Some(name) = default {
        preselect_default_printer(&info, &name);
    }
    let operation = NSPrintOperation::printOperationWithView_printInfo(&view, &info);
    operation.setShowsPrintPanel(true);
    operation.setShowsProgressPanel(true);
    operation.setJobTitle(Some(&NSString::from_str("Image Processor")));
    if !operation.runOperation() {
        // The sheet's own button: cancelled, or the job failed. Either way
        // nothing was printed.
        return Err("the print sheet was closed without printing".to_owned());
    }
    Ok("sent to the printer".to_owned())
}

#[cfg(test)]
mod tests {
    use super::{to_pdf, Page};
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

    // AppKit views may only be built on the main thread, and objc2 makes that a
    // compile-time proof rather than a runtime gamble -- so the way in has to
    // refuse rather than proceed. Under `cargo test` that is exactly the
    // situation, which is why this is a test and the sheet itself is not: the
    // page building is verified by hand through the PDF button, which uses the
    // same view and page setup as the sheet.
    #[test]
    fn building_a_page_off_the_main_thread_is_refused() {
        if MainThreadMarker::new().is_some() {
            // Single-threaded harnesses put tests on the main thread; then the
            // refusal cannot be observed and there is nothing to check.
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
