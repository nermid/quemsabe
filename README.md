# Photographic compositing

A local, non-generative Python workflow. OpenCV GrabCut segments an initial rectangle; PyMatting closed-form alpha estimation and multilevel foreground colour estimation refine transparency and reduce old-background contamination. These are conventional numerical algorithms, without model weights, diffusion, face enhancement, smoothing or invented texture. Colour correction and resampling necessarily change pixel values; identity and detail are not regenerated.

## Setup

Python 3.12:

```sh
cd /workspace/quemsabe
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Place the original photographs at `subject.jpg` and `background.jpg`. They are ignored by Git; do not commit private photos. EXIF orientation is applied; EXIF/GPS metadata is omitted from exports. Convert unusual colour profiles to sRGB in your photo editor first; this workflow assumes sRGB input. No resizing of the background occurs. Keep the largest available originals; the displayed attachments are 768 × 1024.

## First extraction

```sh
.venv/bin/python composite.py --rect 1 290 550 734
```

The example rectangle fits the displayed 768 × 1024 portrait only. Coordinates are in oriented subject pixels. GrabCut treats pixels outside the rectangle as background, so include every visible hair and clothing pixel. For this portrait, the jacket reaches the left and bottom frame boundaries: a manually painted trimap is preferable to rectangle segmentation, which can wrongly remove these edges. Do not reconstruct cropped clothing.

## Refine hair and clothing

1. Inspect `output/edge-review.png` at 100–200%: it shows the cutout against black, white and a checkerboard. Also inspect `composite.png` at native resolution.
2. Open `output/trimap.png` in a pixel editor. Paint solid white (255) only over certain subject, black (0) over certain background, and grey (128) over uncertain boundaries and fine hair. Keep the image dimensions unchanged. Extend uncertain bands around missed hair; disconnected missing strands cannot be recovered from an initial mask that labels them definite background. Save as `refined-trimap.png` outside the output directory.
3. Rerun with the revised trimap and existing parameters:

```sh
.venv/bin/python composite.py --trimap refined-trimap.png --params output/parameters.json --out output-v2
```

4. Compare versions. Repeat until hair, shoulders, lapels and frame boundaries look convincing. Matting is an estimate: if hair colour and old background are indistinguishable, use a different source rather than inventing detail.

`parameters.json` controls subject centre `x,y` in background pixels, uniform `scale`, counterclockwise `rotation` in degrees, brightness/contrast/saturation multipliers, and RGB white-balance gains. Neutral values are 1 (rotation 0). Use small adjustments, e.g. `rgb_gain: [1.02,1.0,0.98]` warms the subject. No automatic lighting match is claimed. Compare face and shirt highlights against the nearby scene; preserve skin texture and avoid clipped white shirt detail. A scale above 1 enlarges existing samples without recovering detail. The tool does not sharpen, smooth skin, blur backgrounds or reshape anatomy.

## Outputs

- `subject.png`: original-size transparent, edge-decontaminated subject before global grading.
- `mask.png`: 8-bit alpha mask; `trimap.png`: editable segmentation constraints.
- `composite.png`: lossless composite at the background resolution.
- `portrait.jpg`: quality 98 JPEG with no chroma subsampling, same dimensions.
- `parameters.json`: editable grading/placement; `provenance.json`: source paths, dimensions and extraction settings.
- `edge-review.png`: full-resolution edge inspection sheet.

Keep originals, trimap and JSON together for edits. Output folders are ignored by Git. Exports intentionally contain no source EXIF metadata.

## This photograph pairing

The supplied background is an indoor market, not an outdoor scene. It is photographed from the top of stairs, looking down. The portrait has a close, roughly eye-level camera and soft window lighting. Place the subject near the camera at the stair landing, with the cropped lower torso reaching the bottom frame. Do not position this cropped portrait as a full person on the stairs. Keep head proportions unchanged. A small rotation adjusts framing only; it cannot correct camera perspective or lighting direction.

The bright roof and mixed warm interior lights may not match the portrait's window illumination. Global colour/brightness adjustments cannot create missing directional light, shadows or reflections. If the placement appears physically implausible, use a new background with eye-level perspective and soft diffuse lighting, or retake the portrait at the location. This tool cannot manufacture cast shadows or restore missing body pixels.
