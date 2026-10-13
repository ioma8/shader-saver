# Undo, redo and export

Two features that both exist to make a shoot's worth of work safe: undo to
experiment freely, export to get the results out.

## Undo and redo

`⌘/Ctrl+Z` steps back, `⌘/Ctrl+Shift+Z` (or `Ctrl+Y`) steps forward. The history
is **per image and in memory**: each image keeps its own stack (a bounded number
of them), but quitting does not keep any of it.

### Where steps come from

Every place that persists an edit already funneled through `save_edits`, so the
history is committed at exactly those points (`commit_history`). The consequences
are worth stating, because they are the reason the design is this boring:

- **no per-frame state comparison** and no pre-change snapshots — the entry at
  the cursor is the state on screen, entries behind it are the past;
- **anything that can change an image is undoable for free**, including Reset All
  Edits, AUTO, Develop RAW and the retouch layer, without any of them knowing
  undo exists;
- **undo is state, so it is exact.** There is no inverse-operation bookkeeping to
  get wrong.

Stepping the history re-runs the whole state through `load_edit_state`, which
re-develops a RAW when `raw_isp_enabled` says the input came from one — undoing
across a Develop RAW has to restore the *input*, not just sliders.

### One step per gesture

A slider drag arrives as one change per frame. A burst of commits inside
`HISTORY_MERGE_SECS` keeps **replacing the entry it began** rather than filling
the stack with intermediate values, so a drag is one step. Three further rules,
each with a test:

- the entry at index 0 is the state the image was opened with, and is never
  merged away;
- editing after an undo starts a fresh step instead of merging backwards, and
  drops the redo branch;
- **a commit that changes nothing is not a step.** Controls that set the dirty
  flag without changing anything (re-selecting the preset already chosen) would
  otherwise leave the user pressing undo and watching nothing happen.

The stack is capped at `HISTORY_STEPS` per image and `HISTORY_IMAGES` images. A
state can carry a 33-cubed look LUT (~860 KB for the pair), so the caps are what
keep a session bounded; realistic use is orders of magnitude smaller.

## Export

`Export…` in the panel, or `⌘/Ctrl+E`. The dialog offers:

- **JPEG** at a chosen quality, or **16-bit PNG** — the first is what a client or
  a print shop wants, the second keeps the depth of the edit chain;
- an optional **longest side** (the aspect is kept), with the output size shown
  before you commit;
- a **resolution** to record in the file, in dpi — a JPEG that says nothing
  about its size leaves whoever prints it to guess, so 300 is the default and 0
  records nothing;
- **Export this image…** to a file, or **Export N picked…** (picks, or with none
  everything the browse filter is showing) to a folder.

What is written is **the crop**, at the size the dialog reports, because both go
through `crop_px` — the same whole-pixel rect the panel draws. A failed write
removes its own half-written file rather than leaving a truncated image at the
target.

### The folder export

Each image is rendered from **its own saved edits**, not its camera preview, so
the batch writes what the editor would show. That means driving the one
`Processor` through other people's images, which has three consequences the code
is built around:

- **one image at a time.** An `ExportJob` carries a full-frame 16-bit readback —
  around 190 MB for a 24 MP frame — so the next is only started once the previous
  file is written. Letting them stack up is how a folder export would run out of
  memory. One *per frame*, so the window keeps drawing.
- **nothing is persisted while it runs.** The processor is holding one of the
  batch's images, so saving the state on screen would write that image's settings
  over the open image's, and push them onto its undo stack. The frame loop skips
  the save and the history commit while a batch is in flight, and the panel says
  editing is paused.
- **the open image is put back afterwards**, from its *saved* state — not from
  the processor, which by then holds the last exported image.

Targets are resolved before anything is written, so two files sharing a stem in
one folder (a RAW and its JPEG, say) get `DSC0001.jpg` and `DSC0001-2.jpg`
instead of quietly overwriting each other.

`Stop` clears the rest of the queue; whatever is already in flight still lands.
