"""Non-generative photographic extraction, matting and compositing."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


def load(path):
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')


def resolve_inputs(paths, subject=None, background=None, input_dir=None):
    """Resolve arbitrary names without guessing which photograph is the person."""
    if paths:
        if len(paths) != 2 or subject or background or input_dir:
            raise ValueError('Provide two paths in subject/background order, or use named options')
        subject, background = paths
    elif input_dir:
        candidates = []
        for path in sorted(Path(input_dir).iterdir()):
            if path.is_file():
                try:
                    with Image.open(path) as im:
                        im.verify()
                    candidates.append(path.resolve())
                except (OSError, ValueError):
                    continue
        def select(value):
            if value is None:
                return None
            path = Path(value)
            return (path if path.is_absolute() else Path(input_dir)/path).resolve()
        subject, background = select(subject), select(background)
        for selected in (subject, background):
            if selected is not None and selected not in candidates:
                raise ValueError(f'Not a readable image in the input directory: {selected}')
        if len(candidates) == 2 and (subject is None) != (background is None):
            remaining = next(p for p in candidates if p != (subject or background))
            subject, background = subject or remaining, background or remaining
        if not subject or not background:
            raise ValueError('Specify --subject and/or --background to assign roles; '
                             'with exactly two images, one role is enough. Found: '
                             + ', '.join(p.name for p in candidates))
    else:
        subject = subject or 'subject.jpg'
        background = background or 'background.jpg'
    if Path(subject).resolve() == Path(background).resolve():
        raise ValueError('Subject and background must be different files')
    for path in (subject, background):
        if not Path(path).is_file():
            raise ValueError(f'Image file not found: {path}. Pass any filenames as '
                             'SUBJECT BACKGROUND; chat image previews are not local files.')
    return str(subject), str(background)


def extract(image, rect, trimap_path=None, edge_width=4):
    from pymatting import estimate_alpha_cf, estimate_foreground_ml
    rgb = np.asarray(image)
    if trimap_path:
        tri = np.asarray(Image.open(trimap_path).convert('L'))
        if tri.shape != rgb.shape[:2]:
            raise ValueError('Trimap must match the EXIF-oriented subject dimensions')
        tri = np.where(tri < 64, 0., np.where(tri > 191, 1., .5))
    else:
        if rect is None:
            raise ValueError('Supply --rect x y width height or a manually painted --trimap')
        x, y, w, h = rect
        if min(x,y) < 0 or min(w,h) <= 0 or x+w > image.width or y+h > image.height:
            raise ValueError('Rectangle must fit inside the subject image')
        mask = np.zeros(rgb.shape[:2], np.uint8)
        cv2.grabCut(rgb, mask, tuple(rect), np.zeros((1,65)), np.zeros((1,65)), 8, cv2.GC_INIT_WITH_RECT)
        binary = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
        kernel = np.ones((2*edge_width+1, 2*edge_width+1), np.uint8)
        inside = cv2.erode(binary, kernel)
        outside = cv2.dilate(binary, kernel)
        tri = np.where(inside == 1, 1., np.where(outside == 0, 0., .5))
    if not np.any(tri == 1) or not np.any(tri == 0):
        raise ValueError('Trimap needs both definite foreground and background')
    alpha = estimate_alpha_cf(rgb.astype(float)/255, tri)
    # Classical foreground estimation removes old-background spill; no learned detail.
    foreground = estimate_foreground_ml(rgb.astype(float)/255, alpha)
    # Preserve original opaque pixels exactly.
    foreground[tri == 1] = rgb[tri == 1]/255
    return np.uint8(np.clip(foreground*255,0,255)), alpha, tri


def grade(rgb, p):
    im = Image.fromarray(rgb)
    for key, cls in [('brightness', ImageEnhance.Brightness), ('contrast', ImageEnhance.Contrast), ('saturation', ImageEnhance.Color)]:
        im = cls(im).enhance(p[key])
    a = np.asarray(im).astype(float) / 255
    # RGB gains are explicit white-balance controls, not inferred relighting.
    return np.clip(a * np.array(p['rgb_gain']), 0, 1)


def render(background, rgb, alpha, p):
    scale = p['scale']
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('scale must be positive and finite')
    h,w = alpha.shape
    matrix = cv2.getRotationMatrix2D((w/2,h/2), p['rotation'], scale)
    matrix[:,2] += np.array([p['x']-w/2, p['y']-h/2])
    size = background.size
    # Resample premultiplied colour and alpha together to avoid dark fringes.
    fg = grade(rgb,p) * alpha[:,:,None]
    warped = cv2.warpAffine(fg, matrix, size, flags=cv2.INTER_LANCZOS4)
    a = np.clip(cv2.warpAffine(alpha, matrix, size, flags=cv2.INTER_LANCZOS4),0,1)
    warped = np.clip(warped,0,a[:,:,None])
    bg = np.asarray(background).astype(float)/255
    # Blend in linear light, retaining the source's photographic texture.
    def linear(v):
        return np.where(v <= .04045, v/12.92, ((v+.055)/1.055)**2.4)
    straight = np.divide(warped,a[:,:,None],out=np.zeros_like(warped),where=a[:,:,None]>1e-8)
    result = linear(straight)*a[:,:,None] + linear(bg)*(1-a[:,:,None])
    result = np.where(result <= .0031308, result*12.92, 1.055*np.maximum(result,0)**(1/2.4)-.055)
    return Image.fromarray(np.uint8(np.clip(result*255,0,255)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='*', help='Any two image paths, in subject/background order')
    parser.add_argument('--subject', help='Subject image path (any filename)')
    parser.add_argument('--background', help='Background image path (any filename)')
    parser.add_argument('--input-dir', help='Folder of images; select one role when it contains exactly two')
    parser.add_argument('--out', default='output')
    parser.add_argument('--rect', nargs=4, type=int)
    parser.add_argument('--trimap')
    parser.add_argument('--params')
    parser.add_argument('--edge-width', type=int, default=4)
    args = parser.parse_args()
    if args.edge_width < 1:
        parser.error('--edge-width must be positive')
    try:
        args.subject, args.background = resolve_inputs(args.images, args.subject, args.background, args.input_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    subject, background = load(args.subject), load(args.background)
    p = dict(x=background.width/2, y=background.height/2, scale=1., rotation=0., brightness=1., contrast=1., saturation=1., rgb_gain=[1.,1.,1.])
    if args.params:
        p.update(json.loads(Path(args.params).read_text()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rgb,alpha,tri = extract(subject,args.rect,args.trimap,args.edge_width)
    Image.fromarray(np.uint8(tri*255)).save(out/'trimap.png')
    Image.fromarray(np.uint8(np.clip(alpha*255,0,255))).save(out/'mask.png')
    Image.fromarray(np.dstack([rgb,np.uint8(alpha*255)])).save(out/'subject.png')
    final = render(background,rgb,alpha,p)
    final.save(out/'composite.png')
    final.save(out/'portrait.jpg',quality=98,subsampling=0)
    # Native-resolution inspection sheets: white, black and checkerboard backgrounds.
    yy,xx = np.indices(alpha.shape)
    checker = np.where(((xx//16+yy//16)%2)[:,:,None],.75,.35)
    panels = [rgb/255*alpha[:,:,None]+b*(1-alpha[:,:,None]) for b in [0.,1.,checker]]
    Image.fromarray(np.uint8(np.clip(np.concatenate(panels,axis=1)*255,0,255))).save(out/'edge-review.png')
    (out/'parameters.json').write_text(json.dumps(p,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(subject=str(Path(args.subject).resolve()),background=str(Path(args.background).resolve()),rect=args.rect,trimap=args.trimap,edge_width=args.edge_width,subject_size=subject.size,background_size=background.size),indent=2)+'\n')
    print(f'Saved native-resolution outputs to {out}; inspect edges before using portrait.jpg.')

if __name__ == '__main__':
    main()
