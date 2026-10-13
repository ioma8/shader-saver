# Printing

`Print…` beside `Export…` opens a print dialog: printer, paper, orientation,
quality, copies, fit-to-page, and a preview of the sheet with the crop on it.
Printing writes the same thing the export does — **the crop** — through the same
render path, then hands the file to CUPS.

## Why CUPS and not the system print dialog

macOS's own print sheet belongs to `NSPrintOperation`, which means Objective-C
FFI, an offscreen `NSView` to render into, and a modal loop running inside a
winit frame. This app is pure Rust with no `unsafe` and no native bindings, and
adding a modal AppKit session underneath its event loop is a good way to get a
hang that cannot be debugged from the outside.

So the dialog is built in egui and the job goes out through `lp`, CUPS's own
command line client, which is exactly what the system sheet ends up doing. What
that buys and costs:

- **You get**: printer choice, the printer's own paper sizes, orientation,
  quality, copies, fit-to-page, and a preview of what will come out.
- **You do not get**: the system preview, a Save-as-PDF destination, or the
  driver-specific panels (colour matching, duplex, tray). A page setup the
  dialog does not cover can still be set as the printer's own default, and
  `Printer default` leaves it alone.

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
