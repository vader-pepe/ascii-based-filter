use std::path::PathBuf;

use clap::Parser;
use image::{GrayImage, ImageReader, Luma, Rgba, RgbaImage, imageops};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input image path
    input: PathBuf,

    /// Output image path
    #[arg(short, long, default_value = "result.png")]
    output: PathBuf,
}

fn load_tiles(path: &str) -> anyhow::Result<[RgbaImage; 10]> {
    let img = image::open(path)?.to_rgba8();
    let (sheet_w, sheet_h) = img.dimensions();
    let tile_w = sheet_w / 10;
    let tile_h = sheet_h;

    let mut tiles: [RgbaImage; 10] = std::array::from_fn(|_| RgbaImage::new(tile_w, tile_h));

    for i in 0..10 {
        let x = i as u32 * tile_w;
        // Crop returns an ImageBuffer view – clone it into a new RgbaImage
        let sub = imageops::crop(&mut img.clone(), x, 0, tile_w, tile_h).to_image();
        tiles[i] = sub;
    }

    Ok(tiles)
}

fn pixelate(img: &RgbaImage, block: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    let mut out = RgbaImage::new(w, h);

    for by in (0..h).step_by(block as usize) {
        for bx in (0..w).step_by(block as usize) {
            let mut sum = [0u64; 4]; // four u64 accumulators
            let mut cnt = 0u64;

            for y in by..(by + block).min(h) {
                for x in bx..(bx + block).min(w) {
                    let p = img.get_pixel(x, y).0;
                    for i in 0..4 {
                        sum[i] += p[i] as u64;
                    }
                    cnt += 1;
                }
            }

            let avg = [
                (sum[0] / cnt) as u8,
                (sum[1] / cnt) as u8,
                (sum[2] / cnt) as u8,
                (sum[3] / cnt) as u8,
            ];

            let avg_px = Rgba(avg);

            for y in by..(by + block).min(h) {
                for x in bx..(bx + block).min(w) {
                    out.put_pixel(x, y, avg_px);
                }
            }
        }
    }
    out
}

/// Desaturate an RGBA image in place by blending channels toward luminance.
/// `amount`: 0.0 = no change, 1.0 = fully desaturated (all gray).
fn desaturate_in_place(img: &mut RgbaImage, amount: f32) {
    let (w, h) = img.dimensions();

    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel_mut(x, y);
            let [r, g, b, a] = p.0;

            // Perceptual luminance ‑ use BT.709 or BT.601 coefficients
            let lum = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) as u8;

            // Interpolate channels toward luminance
            let blend = |orig: u8| -> u8 {
                ((1.0 - amount) * orig as f32 + amount * lum as f32).round() as u8
            };

            *p = Rgba([blend(r), blend(g), blend(b), a]);
        }
    }
}

/// Quantizes a luminance value (0.0..=1.0) to 10 discrete levels: 0.0, 0.1, …, 0.9
fn quantize_luminance(f: f32) -> f32 {
    let x = f.clamp(0.0, 1.0); // clamp ensures f stays in [0,1] :contentReference[oaicite:1]{index=1}
    (x * 10.0).floor() / 10.0
}

pub fn quantize_image(img: &GrayImage) -> GrayImage {
    let mut out = img.clone();

    for px in out.pixels_mut() {
        let Luma([y]) = *px;
        let lum = f32::from(y) / 255.0;
        let qlum = quantize_luminance(lum);
        let qy = (qlum * 255.0).round() as u8;
        *px = Luma([qy]);
    }

    out
}

/// Produces a block-level index image where each 8×8 block's top-left pixel is the quantized index.
pub fn quantize_blocks_to_indices(src: &GrayImage) -> GrayImage {
    let (w, h) = src.dimensions();
    let bw = 8;
    let bx = w / bw;
    let by = h / bw;
    let mut idx_img = GrayImage::new(bx, by);

    for by_i in 0..by {
        for bx_i in 0..bx {
            let first = src.get_pixel(bx_i * bw, by_i * bw)[0];
            // Force full range 0..=255 to map exactly into 0..=9
            let idx = ((first as u32 * 10 + 255) / 256).min(9) as u8;
            idx_img.put_pixel(bx_i, by_i, Luma([idx as u8]));
        }
    }

    idx_img
}

/// Replace each 8×8 block in the original size using RGBA tiles
pub fn replace_blocks_with_tiles(
    indices: &GrayImage, // dims: w/8 × h/8
    tiles: &[RgbaImage; 10],
) -> RgbaImage {
    let (bx, by) = indices.dimensions();
    let bw = 8;
    let mut out = RgbaImage::new(bx * bw, by * bw);

    for by in 0..by {
        for bx in 0..bx {
            let idx = indices.get_pixel(bx, by)[0] as usize;
            assert!(idx < 10, "Invalid block index {}", idx);
            let tile = &tiles[idx];
            for ty in 0..bw {
                for tx in 0..bw {
                    let p = tile.get_pixel(tx, ty);
                    out.put_pixel(bx * bw + tx, by * bw + ty, *p);
                }
            }
        }
    }

    out
}

fn _convert_to_rgba(gray: &GrayImage) -> RgbaImage {
    RgbaImage::from_fn(gray.width(), gray.height(), |x, y| {
        let pixel = gray.get_pixel(x, y);
        let value = pixel.0[0];
        Rgba([value, value, value, 255])
    })
}

fn main() -> image::ImageResult<()> {
    let args = Args::parse();

    // Load input image
    let image_source = ImageReader::open(&args.input)?.decode()?.to_rgba8();
    println!(
        "Loaded image: {}x{}",
        image_source.width(),
        image_source.height()
    );

    let filling_tiles = load_tiles("fillASCII.png").unwrap();
    let mut image_downscaled = pixelate(&image_source, 8);
    desaturate_in_place(&mut image_downscaled, 1.0);
    let image_greyscaled = image::imageops::grayscale(&image_downscaled);
    let image_quantized = quantize_image(&image_greyscaled);
    let image_indices = quantize_blocks_to_indices(&image_quantized);
    let result = replace_blocks_with_tiles(&image_indices, &filling_tiles);

    result.save("result.png")?;
    Ok(())
}
