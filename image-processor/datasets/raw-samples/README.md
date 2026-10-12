# RAW sample set

RAW files from many manufacturers, used to check that the universal develop in
`src/raw_develop.rs` works on cameras and formats it was never tuned on. They
are **not committed** (see `.gitignore`); only this README is tracked.

18 files, ~620 MB.

## Source and license

All files come from [raw.pixls.us](https://raw.pixls.us/), which publishes the
samples under **[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)**
(public domain dedication). None of these cameras are in that site's non-CC0
list. Credit: the raw.pixls.us contributors and the darktable/rawspeed
communities.

## Files

| File | Camera | Format | Size |
| --- | --- | --- | --- |
| `apple/iphone6s.dng` | Apple iPhone 6s Plus | DNG | 10 MB |
| `canon/eos-r5m2.cr3` | Canon EOS R5 Mark II | CR3 | 52 MB |
| `dji/osmo-action.dng` | DJI Osmo Action | DNG | 24 MB |
| `epson/rd1.erf` | Epson R-D1 | ERF | 10 MB |
| `fujifilm/xt3.raf` | Fujifilm X-T3 (X-Trans) | RAF | 53 MB |
| `google/pixel2xl.dng` | Google Pixel 2 XL | DNG | 23 MB |
| `hasselblad/x1d.3fr` | Hasselblad X1D | 3FR | 105 MB |
| `kodak/dcs760c.dcr` | Kodak DCS760C | DCR | 7 MB |
| `leica/m10.dng` | Leica M10 | DNG | 33 MB |
| `minolta/dimage7.mrw` | Minolta DiMAGE 7 | MRW | 10 MB |
| `nikon/d850.nef` | Nikon D850 | NEF | 37 MB |
| `olympus/e1.orf` | Olympus E-1 | ORF | 10 MB |
| `omdigital/om-3.orf` | OM System OM-3 | ORF | 21 MB |
| `panasonic/gx800.rw2` | Panasonic DC-GX800 | RW2 | 19 MB |
| `pentax/kr.pef` | Pentax K-r | PEF | 13 MB |
| `phaseone/iq3.iiq` | Phase One IQ3 100MP | IIQ | 104 MB |
| `samsung/nx300.srw` | Samsung NX300 | SRW | 22 MB |
| `sony/a1.arw` | Sony A1 | ARW | 60 MB |

Two files catch decoder bugs rather than just exercising coverage:

- The **Fuji X-T3** uses an X-Trans CFA (not Bayer). It exposed a demosaic bug
  fixed by moving to `rawler 0.8` (see
  [`docs/raw-development.md`](../../docs/raw-development.md)).
- The **Samsung NX300** and **Phase One IQ3** exercised formats (SRW, IIQ) that
  the app's extension list did not accept, so they were invisible to the app
  until the list was widened.

## Run the check

```bash
RAW_DATASET_DIR=datasets/raw-samples cargo test --release \
  dataset_samples_develop_and_are_sane -- --nocapture
```

It develops every file, prints exposure/clipping, asserts nothing clips badly,
and writes PNGs to `/tmp/raw-dataset` for a visual pass.

MAE is a separate bench that only uses files with a **usable** reference
(sidecar JPEG, else a decent embedded JPEG; tiny thumbnails and black previews
are skipped):

```bash
RAW_DATASET_DIR=datasets/raw-samples cargo test --release \
  reference_bench_runs -- --nocapture
```

A reference is unusable when it is smaller than 512 px on the long edge or
extracts black. In this set that is: Canon CR3 (all previews extract black),
Fujifilm X-T3 (160×120), Panasonic GX800 (160×120), DJI Osmo Action (192×144),
and Leica M10 (no embedded JPEG). The remaining 13 files give **mean MAE 0.1182,
median 0.0833** against their own camera previews.

The reference is the **embedded camera JPEG**, not a true SOOC file. That is a
good proxy for tone and color but carries the camera's picture style
(Fujifilm film simulations, Canon Picture Styles, ...), so a nonzero MAE is
expected and does not by itself mean the develop is wrong.

## Known limitations found by this set

| Sample | Problem | Cause |
| --- | --- | --- |
| Kodak DCS760C | Develop has a yellow-green cast | Old (2001) DCR; likely no usable as-shot white balance in the file |
| Canon CR3 | Embedded preview extracts black | `jpgfromraw` limitation; MAE skipped |
| Sigma (X3F) | Not included | `rawler` fails: `Couldn't find tag TiffCommonTag::Make` |
| GoPro (GPR) | Not included | `rawler` fails: `TODO: Unsupported DNG compression` |
| Samsung EK-GN120 | Not included | `rawler`: `Unknown camera`; NX300 used instead |

## Re-download

```bash
cd image-processor
base=https://raw.pixls.us/data
mkdir -p datasets/raw-samples/{apple,canon,dji,epson,fujifilm,google,hasselblad,kodak,leica,minolta,nikon,olympus,omdigital,panasonic,pentax,phaseone,samsung,sony}
curl -sL -o datasets/raw-samples/apple/iphone6s.dng      "$base/Apple/iPhone%206s%20Plus/IMG_0853.DNG"
curl -sL -o datasets/raw-samples/canon/eos-r5m2.cr3      "$base/Canon/Canon%20EOS%20R5m2/RAW.CR3"
curl -sL -o datasets/raw-samples/dji/osmo-action.dng     "$base/DJI/DJI%20Osmo%20Action/DJI_0254.DNG"
curl -sL -o datasets/raw-samples/epson/rd1.erf           "$base/Epson/R-D1/_EPS0672.ERF"
curl -sL -o datasets/raw-samples/fujifilm/xt3.raf        "$base/Fujifilm/X-T3/AFXT2720.RAF"
curl -sL -o datasets/raw-samples/google/pixel2xl.dng     "$base/Google/Pixel%202%20XL/IMG_20180109_080629.dng"
curl -sL -o datasets/raw-samples/hasselblad/x1d.3fr      "$base/Hasselblad/X1D/B9999910.3FR"
curl -sL -o datasets/raw-samples/kodak/dcs760c.dcr       "$base/Kodak/DCS760C/86L57188.DCR"
curl -sL -o datasets/raw-samples/leica/m10.dng           "$base/Leica/M10/f5381888.dng"
curl -sL -o datasets/raw-samples/minolta/dimage7.mrw     "$base/Minolta/DiMAGE%207/RAW_MINOLTA_DIMAGE7.MRW"
curl -sL -o datasets/raw-samples/nikon/d850.nef          "$base/Nikon/D850/Nikon-D850-12bit-compressed.NEF"
curl -sL -o datasets/raw-samples/olympus/e1.orf          "$base/Olympus/E-1/E_1__C106743_gredos.ORF"
curl -sL -o datasets/raw-samples/omdigital/om-3.orf      "$base/OM%20Digital%20Solutions/OM-3/P5048182.ORF"
curl -sL -o datasets/raw-samples/panasonic/gx800.rw2     "$base/Panasonic/DC-GX800/P1000091.RW2"
curl -sL -o datasets/raw-samples/pentax/kr.pef           "$base/PENTAX/PENTAX%20K-r/IMGP4424.PEF"
curl -sL -o datasets/raw-samples/phaseone/iq3.iiq        "$base/Phase%20One/IQ3%20100MP/Phase_One_IQ3_100MP_IIQ_L.IIQ"
curl -sL -o datasets/raw-samples/samsung/nx300.srw       "$base/Samsung/NX300/nx300.SRW"
curl -sL -o datasets/raw-samples/sony/a1.arw             "$base/Sony/ILCE-1/A1_full_compressed.ARW"
```
