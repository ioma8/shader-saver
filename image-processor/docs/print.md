# Printing

`Print…` in the right panel prints the crop. Two implementations, chosen by
platform, because there is no cross-platform print dialog to use:

- **macOS** uses AppKit's `NSPrintOperation` — the real system sheet, with the
  printer, presets, paper size, *paper type*, colour matching and everything
  else the driver offers, plus Save-as-PDF and Preview. `PDF…` beside it writes
  a print-ready PDF without touching a printer.
- **Other platforms** use the CUPS dialog described further down: the app's own
  options panel and `lp`.

The image is the same either way: **the crop**, rendered once through the same
export path, with its resolution recorded in the file.

## macOS: the system sheet

This is the better route and it is the one the app takes on a Mac, for one
concrete reason: printing a photo needs settings that live in the *driver*, and
a hand-built dialog can only send the ones it knows about. The paper type is the
one that bit us (below); the sheet covers that and every other.

How it works, in `src/print_macos.rs`:

- the rendered JPEG becomes an `NSImage`, which goes into a **stock
  `NSImageView`** — no view subclass, no message forwarding;
- its frame is set to the image's own size, which AppKit derives from the
  resolution recorded in the file. That is what the render's `dpi` is for: a
  1200 px render at 600 dpi is two inches, and a test pins exactly that, since
  it is the one part of the page geometry that can be checked without the main
  thread;
- page setup is `NSPrintingPaginationMode::Fit`, so the image scales to whatever
  paper the sheet is on;
- `NSPrintInfo::new()` picks up the **shared printer settings**, so the sheet
  opens on the paper and printer last used — on the machine this was written
  for, the 4×6 photo paper from the previous print;
- the printer CUPS calls default is preselected, best effort.

What is `unsafe` is two class-method calls. Everything else goes through objc2,
whose bindings are memory-safe and use `MainThreadMarker` to *prove*
main-thread access at compile time — AppKit views may not be built anywhere
else. `to_pdf` refuses with a plain error rather than risking it, and a test
pins that refusal.

### What is checked, and what is not

Checked by hand, because it is the whole risky half: a real photograph goes
through the same view and page setup and comes out as a 1.87 MB PDF of one
page, which rasterises to the photograph, correctly scaled to the paper
(310.8 × 466.2 pt — the imageable area of 4×6 photo paper, because that is what
the printer was set to). `PDF…` is the same code path, so it doubles as the
"look before you spend paper" button.

**Not** checked here: the sheet itself. `cargo test` runs tests on spawned
threads, and AppKit views require the main thread, so no test can put the sheet
up — the first print from this build is the first real exercise of it. The
modal-loop hazard is real and known: a sheet pumps the run loop, which is why
the render happens on a worker and only the sheet is modal, and why this is the
same pattern the app already uses for file dialogs (`rfd`), which are also
modal AppKit panels opened from inside the frame.

## The CUPS path (everything that is not macOS)

There is no print-dialog crate to reach for, and no cross-platform C library
either: a print dialog belongs to a GUI toolkit (AppKit here, `PrintDlgExW` on
Windows, GTK or Qt on Linux), while CUPS itself is a spooler with no UI. The
nearest Rust crate, `printers`, is documented as running `lp`/`lpstat` on unix —
which is exactly what this path does.

So where there is no system sheet the dialog is egui's and the job goes out
through `lp`, CUPS's own command line client. What that buys and costs:

- **You get**: printer choice, the printer's own paper sizes and *paper types*,
  orientation, quality, copies, fit-to-page, and a preview of what comes out.
- **You do not get**: the system preview, a Save-as-PDF destination, or the
  driver panels the sheet would show (colour matching, duplex, tray).

## The paper type is what decides the ink

This is the setting that matters most, and leaving it out is not neutral. A CUPS
queue for a photo printer offers `MediaType`, defaulting to `auto` — which means
*whatever the printer's own panel says*, and a printer's panel says plain paper.
Plain-paper ink limits on glossy photo paper is a bleached, washed-out print,
which is what prompted this section: a photo printed through this dialog with
nothing but the paper *size* set came out pale.

So the paper type is a control in the dialog, filled from the printer's own
list of types (`com.canon-mtglossy` shows as `glossy`, `photographic` stays as
it is — the labels only take the vendor prefix off, so nothing is invented about
what a printer calls its papers). When it is left on the printer's default the
dialog says why that is risky, in amber, every time.

It is remembered **by value** across sessions rather than by index, because the
paper in the tray is the same paper next time — and a printer-specific option
index means nothing on another printer. The paper *size* is still chosen fresh
each time for the same reason.

## The quality option is spelled the way the queue defines it

The job used to send the IPP attribute name, `print-quality=4`. That queue does
not define an option called `print-quality` — its PPD offers
`cupsPrintQuality/Quality: Draft *Normal High` — so CUPS ignored it, and the
queue's own default was in force the whole time. Both are valid `lp` options and
both reach the printer as the same IPP attribute, but only the second is what a
CUPS queue actually looks up, so that is what is sent now.

## Everything about the paper comes from the printer

A fixed list of sizes would be wrong on the machine this was written on. Its
Canon lists **41** of them: A4 and A5 but **no A3**, Letter and Legal but no
Tabloid, `215x345mm` and `89x89mm` alongside `4x6` and `8x10`, envelopes, and a
custom-size placeholder. A built-in list would offer paper that printer cannot
load and miss paper it can. Sending it a `media` name it does not know fails the
job, so the list is read from the printer's own description
(`lpoptions -p <printer> -l`) and the sizes are labelled from the names:
`4x6` becomes `4 × 6 in`, `215x345mm` stays millimetres, `5x5.Fullbleed` is
`5 × 5 in (borderless)`, and `A4` keeps its name. The printer's own default is
named at the top (`Printer default (A4)`), and the first entry — which passes no
`media` option at all — is always available because it always works.

An entry whose size cannot be read (a printer's custom-size placeholder, say)
keeps its raw name, and the dialog says so rather than showing a sheet it is
guessing at.

## Two things learned by running it

Both are pinned by tests with the real output in them.

**`lpstat` is localised, and `LC_ALL` does not change that.** Apple's CUPS
build takes its messages from the system locale, so on a Czech machine
`lpstat -p` says `tiskárna Canon_G3030_series je nečinná` no matter what the
environment says. A parser keyed on English prose breaks. The parsers here are
positional instead, which works in either language because only the *words*
change:

| query | where the name is |
|---|---|
| `lpstat -a` | the first token — `Canon… přijímá požadavky od …` |
| `lpstat -p` | the second — `tiskárna Canon… je nečinná` |
| `lpstat -d` | after the colon — `výchozí systémový cíl: Canon…` |

`-a` is the primary source (a name that cannot be confused with prose), with
`-p` adding queues that are stopped and so accepting nothing.

**A size with a decimal point is not a variant.** `.Fullbleed` is stripped by
matching a known suffix, not by splitting on the first dot: `3.5x5` was briefly
read as a variant of a paper called `3`. The test fixture has it now.

## What gets rendered

The crop is rendered once, as a JPEG at quality 92, sized to fit the paper's
pixel box at `PRINT_DPI` (600). That box is a bound on the spool file, **not** a
claim about the printed resolution — the printer scales what it is given onto
the page, and an image smaller than the page is deliberately not enlarged here
(the dialog says so). A 6000 × 4000 frame on A4 goes out as 4961 × 3307.

Margins are the printer's own: `fit-to-page` hands the scaling and the
unprintable border to the driver rather than guessing at them.

The file is written to `~/.image-processor/print.jpg` and reused each time, so
there is nothing to clean up.

## Nothing waits forever

`std::process` has no timeout, so every command goes through a bounded runner:
5 seconds for a query, 60 for `lp`. A scheduler that answers at all answers in
milliseconds (70 ms measured here) — but one that has stopped answering would
otherwise block whichever thread asked. That matters more than it sounds:

- the **queries run on a worker**, not the UI thread, so opening the dialog
  never waits on CUPS at all; while one is in flight the dialog says so and the
  Print button waits. Asking about a different printer replaces the question,
  and the first answer is dropped with its receiver.
- **`lp` is bounded too.** If it never answers, the job is reported as *unknown*
  — "the job may or may not have been sent" — rather than leaving the Print
  button disabled for the rest of the session, which is what waiting forever
  would do.

A failed or timed-out query is treated as an empty one: the dialog falls back to
`System default` and the printer's own default paper, both of which still print.

## It records the size it prepared

The JPEG handed to `lp` carries its resolution in the JFIF header, because a
file that does not say how big it is leaves the driver guessing — and the
guess is what turns "Fit to page" off into an unpredictable result. With the
resolution recorded, turning fitting off means *the size this was prepared at*:
a 4961 px frame laid out for A4 is 8.27 inches at 600, which is the page.

That is also why the export dialog has a **Resolution** field: a JPEG sent to a
client is better off saying what size it is, and 300 dpi is the conventional
default. Zero records nothing, which is the same as before that field existed.

## While an export is running

Printing and single-image exporting both act on the image the processor is
holding — and during a folder export that is one of the *batch's* images, not
the one on screen. Both therefore stand aside while a batch runs, with the
dialog saying why. Nothing is corrupted either way; what is prevented is
printing a different photograph than the one you are looking at.

## What was checked before blaming the printer

A bleached print has two possible homes: the file, or the machine. The file was
ruled out by measurement rather than assumption — render a smooth gradient,
export it through the same job the print path builds, and compare the file
against the display texture: mean 133.57 against 133.08, worst pixel 2 levels
apart, which is JPEG noise on a smooth ramp. A gamma or scaling slip anywhere
between the panel and the encoded file would have shown as a brightness shift
across the whole frame. That check is now a test
(`the_export_matches_what_the_panel_shows`), so it cannot come back.

What the printer reported at the same time, worth knowing rather than acting on:
`marker-levels` showed the four inks at 50% and the maintenance cartridge at 5%
— that is 95% *used*, so it is worth watching, though a spent maintenance
cartridge stops a Canon rather than fading it.

## Failure is shown, not swallowed

`lp` reports an unknown printer, a stopped queue or an unreadable file on
stderr, in the user's own language; the dialog shows that text verbatim, in red,
under the buttons. A print that cannot even be rendered says so too.

## Tests

`src/print.rs` covers the parsers against real Czech *and* English output, the
paper list against the Canon's real options (including the decimal-size
regression), the millimetre conversion rules, the page box and the fitting
rule, and the exact `lp` argument vector for a default job and a fully-specified
one. Nothing in the suite sends a print job.
