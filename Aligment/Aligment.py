# Aligment.py
from PIL import Image
import cv2
import numpy as np
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

def align_image(img, reference=None, reference_path='test_0015_aligned.jpg',
                nfeatures=2000, min_match_count=8, return_type='pil', scale_for_detection=None):
    """
    Allinea `img` verso `reference` (o `reference_path`).
    `img`/`reference` possono essere: path str, PIL.Image, numpy.ndarray (RGB/BGR/gray).
    return_type: 'pil' (default), 'opencv' (BGR numpy), 'rgb_numpy'
    scale_for_detection: riduce temporaneamente la risoluzione per velocizzare il detection (es. 0.5)
    """
    def _load_to_bgr(x):
        if isinstance(x, str):
            bgr = cv2.imread(x)
            if bgr is None:
                raise FileNotFoundError(f"Cannot read image: {x}")
            return bgr
        if isinstance(x, Image.Image):
            return cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            # assume RGB if 3 channels
            return cv2.cvtColor(x, cv2.COLOR_RGB2BGR) if x.shape[2] == 3 else x
        raise TypeError("Unsupported image type")

    img_bgr = _load_to_bgr(img)
    ref_bgr = _load_to_bgr(reference if reference is not None else reference_path)

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Optionally scale down for detection to speed up
    if scale_for_detection is not None and 0 < scale_for_detection < 1.0:
        def _rescale(img, s):
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w * s), int(h * s)))
        img_gray_small = _rescale(img_gray, scale_for_detection)
        ref_gray_small = _rescale(ref_gray, scale_for_detection)
    else:
        img_gray_small, ref_gray_small = img_gray, ref_gray

    orb = cv2.ORB_create(nfeatures)
    kp1, des1 = orb.detectAndCompute(ref_gray_small, None)
    kp2, des2 = orb.detectAndCompute(img_gray_small, None)

    # if descriptors missing, fallback: return original image converted to requested format
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return _fallback(img_bgr, return_type)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in knn_matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
            good.append(m)

    if len(good) < min_match_count:
        return _fallback(img_bgr, return_type)

    # recover keypoints coordinates in original scale (if scaling used)
    if scale_for_detection is not None and 0 < scale_for_detection < 1.0:
        scale = 1.0 / scale_for_detection
        pts_ref = np.float32([ (kp1[m.queryIdx].pt[0]*scale, kp1[m.queryIdx].pt[1]*scale) for m in good ]).reshape(-1,1,2)
        pts_img = np.float32([ (kp2[m.trainIdx].pt[0]*scale, kp2[m.trainIdx].pt[1]*scale) for m in good ]).reshape(-1,1,2)
    else:
        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pts_img = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, mask = cv2.findHomography(pts_img, pts_ref, cv2.RANSAC, 5.0)
    if H is None:
        return _fallback(img_bgr, return_type)

    h_ref, w_ref = ref_gray.shape
    aligned = cv2.warpPerspective(img_bgr, H, (w_ref, h_ref), flags=cv2.INTER_LINEAR)

    if return_type == 'opencv':
        return aligned
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    if return_type == 'rgb_numpy':
        return rgb
    return Image.fromarray(rgb)

def _fallback(img_bgr, return_type):
    if return_type == 'opencv':
        return img_bgr
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if return_type == 'rgb_numpy':
        return rgb
    return Image.fromarray(rgb)



def process_file(src_path, dst_path, ref, min_match_count, scale_for_detection, copy_on_fail):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = align_image(str(src_path), reference=ref, min_match_count=min_match_count, scale_for_detection=scale_for_detection, return_type='pil')
        out.save(dst_path, quality=95)
        return True, str(src_path)
    except Exception as e:
        if copy_on_fail:
            # copy original
            Image.open(src_path).convert('RGB').save(dst_path, quality=95)
            return False, f"{src_path} (copied original due to error: {e})"
        return False, f"{src_path} (failed: {e})"

def main(args):
    src = Path(args.src)
    dst = Path(args.dst)
    assert src.exists()

    # Gather images
    exts = ('*.jpg','*.jpeg','*.png','*.bmp')
    files = []
    for e in exts:
        files += list(src.rglob(e))
    if args.max_files:
        files = files[:args.max_files]

    total = len(files)
    print(f"Found {total} files to process.")

    # load reference image if provided path
    ref = args.reference

    failures = []
    success = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = { ex.submit(process_file, f, dst / f.relative_to(src), ref, args.min_match_count, args.scale_for_detection, args.copy_on_fail): f for f in files }
        for fut in tqdm(as_completed(futures), total=total):
            ok, msg = fut.result()
            if ok:
                success += 1
            else:
                failures.append(msg)

    print(f"Done. Success: {success}/{total}. Failures: {len(failures)}")
    if failures:
        with open(dst / "alignment_failures.log", "w") as fh:
            fh.write("\n".join(failures))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help="AffectNet train/test root (ImageFolder structure)")
    parser.add_argument('--dst', required=True, help="Output root to write aligned images (will preserve subfolders)")
    parser.add_argument('--reference', required=True, help="Reference image path")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--min_match_count', type=int, default=8)
    parser.add_argument('--scale_for_detection', type=float, default=None, help="Optional scale (0-1) to speed up detection, e.g. 0.5")
    parser.add_argument('--copy-on-fail', action='store_true', help="If set, copy original image when alignment fails")
    parser.add_argument('--max-files', type=int, default=0, help="Limit number of files (for testing)")
    args = parser.parse_args()
    if args.max_files == 0:
        args.max_files = None
    main(args)