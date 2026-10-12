    use super::*;

        #[test]
        fn embedded_raw_render_model_is_valid_and_monotone() {
            let model = RawRenderModel::embedded().expect("embedded RAW render model");
            let input: [f32; RENDER_FEATURES] =
                std::array::from_fn(|index| RENDER_QUANTILES[index % RENDER_QUANTILES.len()]);
            let output = model.predict(&input);
            assert!(output.iter().all(|value| value.is_finite()));
            for channel in 0..3 {
                assert!(output
                    [channel * RENDER_QUANTILES.len()..(channel + 1) * RENDER_QUANTILES.len()]
                    .windows(2)
                    .all(|pair| pair[0] <= pair[1]));
            }
        }

        #[test]
        fn u16_develop_matches_8bit_develop() {
            let path =
                Path::new("/Users/jakubkolcar/Downloads/PXL_20260806_114652240.RAW-02.ORIGINAL.dng");
            if !path.exists() {
                return;
            }
            let dev8 = develop_raw(path, 512).unwrap();
            let dev16 = develop_raw_u16(path, 512).unwrap();
            assert_eq!(dev16.dimensions(), dev8.dimensions());
            // u16 output is the same srgb-encoded value at 16-bit resolution.
            let mut checked = 0;
            for (pixel, px16) in dev8.pixels().zip(dev16.pixels()) {
                for channel in 0..3 {
                    let expected = u32::from(pixel[channel]) * 257;
                    let actual = u32::from(px16[channel]);
                    assert!(
                        actual.abs_diff(expected) <= 257,
                        "u16 develop diverges at {channel}: {actual} vs {expected}"
                    );
                }
                assert_eq!(px16[3], 65535);
                checked += 1;
            }
            assert!(checked > 1000);
        }

        #[test]
        fn tone_keeps_neutral_grey_neutral() {
            let mut image = Rgba16Image::from_fn(16, 16, |x, y| {
                let v = if (x + y) % 2 == 0 { 9000 } else { 30000 };
                image::Rgba([v, v, v, 65535])
            });
            tone_u16(&mut image);
            for pixel in image.pixels() {
                assert!((i32::from(pixel[0]) - i32::from(pixel[1])).abs() <= 1);
                assert!((i32::from(pixel[1]) - i32::from(pixel[2])).abs() <= 1);
            }
        }

        #[test]
        fn meter_stays_neutral_and_bounded() {
            let mut flat = Rgba16Image::from_pixel(8, 8, image::Rgba([30_000, 30_000, 30_000, 65535]));
            let raw = raw_features(&flat);
            meter_u16(&mut flat, &raw);
            for p in flat.pixels() {
                assert_eq!(p[0], p[1]);
                assert_eq!(p[1], p[2]);
            }
            let mut dark = Rgba16Image::from_pixel(8, 8, image::Rgba([200, 200, 200, 65535]));
            let raw = raw_features(&dark);
            meter_u16(&mut dark, &raw);
            let p = dark.get_pixel(0, 0);
            assert_eq!(p[0], p[1]);
            assert!(p[0] < 65_535);
        }

        #[test]
        fn s_curve_is_monotone_with_fixed_endpoints() {
            assert_eq!(s_curve(0.0, TONE.contrast), 0.0);
            assert!((s_curve(1.0, TONE.contrast) - 1.0).abs() < 1e-6);
            let mut last = -1.0;
            for i in 0..=100 {
                let v = s_curve(i as f32 / 100.0, TONE.contrast);
                assert!(v >= last);
                last = v;
            }
        }
