import os
import cv2
import numpy as np

# keypoints: left_eye, right_eye, nose, mouth_left, mouth_right
try:
    from facenet_pytorch import MTCNN
    import torch
except Exception as e:
    raise RuntimeError("Install 'facenet-pytorch' and 'torch' (e.g. 'pip install facenet-pytorch torch') to execute this script") from e


class AlignerMtcnn:
    """
    Class that exposes MTCNN-based alignment functions.
    It keeps the same conceptual structure as the original file:
    - method get_face_keypoints_mtcnn(image_bgr) that returns a (5,2) array or None
    - align_to_template(img, out_size=(W,H)) and __call__(img) are useful for plugging it into transforms.

    Accepts NumPy images H \times W \times 3 (RGB or BGR); it uses an internal heuristic to convert and feed the detector in RGB format.

    For use in transforms, __call__ returns an RGB image with size out_size.

    """

    def __init__(self, device=None, keep_all=False, out_size=(224, 224), fallback_to_original=True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=keep_all, device=self.device)
        self.out_size = tuple(out_size)
        self.fallback_to_original = fallback_to_original

    def _to_rgb_for_detector(self, img: np.ndarray) -> np.ndarray:
        """Return RGB image suited for MTCNN. Accept RGB or BGR in input."""
        img = np.asarray(img)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Image needs to be HxWx3")

        # Simple euristich to distinguish between BGR and RGB: compare average of channels
        b_mean = img[:, :, 0].mean()
        r_mean = img[:, :, 2].mean()
        if b_mean > r_mean:
            # probably BGR
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # probably RGB
            return img.copy()

    def get_face_keypoints_mtcnn(self, image_bgr: np.ndarray):
        """Returns an array (5,2) with points (x,y) in float32 or None if it doesn't find faces.

        Keep the original signature of the file (accepts RGB images), but functions also in RGB.

        """
        if image_bgr is None:
            return None

        image_rgb = self._to_rgb_for_detector(image_bgr)

        boxes, probs, landmarks = self.mtcnn.detect(image_rgb, landmarks=True)
        # landmarks: array (n,5,2) in (x,y) format
        if landmarks is None or len(landmarks) == 0:
            return None

        pts = landmarks[0].astype(np.float32)
        return pts

    def compute_affine(self, src_pts: np.ndarray, dst_pts: np.ndarray):
        """Compute the affine transformation (estimateAffinePartial2D) and returns it."""
        if src_pts is None or dst_pts is None:
            return None
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        return M


    def _canonical_dst(self, out_size=None):
        W, H = (out_size or self.out_size)
        return np.array([
            [0.35 * W, 0.35 * H],  # left eye
            [0.65 * W, 0.35 * H],  # right eye
            [0.50 * W, 0.50 * H],  # nose
            [0.37 * W, 0.75 * H],  # mouth left
            [0.63 * W, 0.75 * H],  # mouth right
        ], dtype=np.float32)

    def align_to_template(self, img_src: np.ndarray, out_size=None):
        if img_src is None:
            return None

        out_size = out_size or self.out_size
        kp_src = self.get_face_keypoints_mtcnn(img_src)
        if kp_src is None:
            if self.fallback_to_original:
                resized = cv2.resize(img_src, (out_size[0], out_size[1]), interpolation=cv2.INTER_LINEAR)

                rgb = self._to_rgb_for_detector(resized)
                return rgb
            else:
                raise RuntimeError("No face detected in the image")

        dst = self._canonical_dst(out_size)
        M = self.compute_affine(kp_src, dst)
        if M is None:
            if self.fallback_to_original:
                resized = cv2.resize(img_src, (out_size[0], out_size[1]), interpolation=cv2.INTER_LINEAR)
                rgb = self._to_rgb_for_detector(resized)
                return rgb
            else:
                raise RuntimeError("Impossible to compute the affine transformation for the alignment")


        img_rgb = self._to_rgb_for_detector(img_src)
        aligned = cv2.warpAffine(img_rgb, M, (out_size[0], out_size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return aligned

    def __call__(self, img):
        return self.align_to_template(img)


if __name__ == "__main__":
    aligner = AlignerMtcnn()

    img_src_path = "Dataset/image0000416.jpg"

    img_src = cv2.imread(img_src_path)

    if img_src is None:
        raise RuntimeError(f"img_src = None (wrong path or non readable file): {img_src_path}")

    aligned = aligner.align_to_template(img_src, out_size=(224, 224))

    cv2.imshow("Original Source", img_src)
    cv2.imshow("Aligned Source", aligned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

