import os
import cv2
import numpy as np

# Sostituito l'uso di dlib con MTCNN (facenet-pytorch) per rilevamento dei volti e dei 5 keypoints
# I keypoints restituiti sono: left_eye, right_eye, nose, mouth_left, mouth_right
try:
    from facenet_pytorch import MTCNN
    import torch
except Exception as e:
    raise RuntimeError("Installa 'facenet-pytorch' e 'torch' (es. 'pip install facenet-pytorch torch') per eseguire questo script") from e


class AlignerMtcnn:
    """Classe che espone le funzioni di allineamento basate su MTCNN.

    Mantiene la stessa concettualità del file originale:
    - metodo `get_face_keypoints_mtcnn(image_bgr)` che ritorna (5,2) o None


    `align_to_template(img, out_size=(W,H))` e `__call__(img)` utili per inserirla nei `transforms`.

    Note:
    - Accetta immagini numpy HxWx3 (RGB o BGR); usa un'euristica interna per passare al detector in formato RGB.
    - Per l'uso nei transforms `__call__` ritorna un'immagine RGB di dimensione `out_size`.
    """

    def __init__(self, device=None, keep_all=False, out_size=(224, 224), fallback_to_original=True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=keep_all, device=self.device)
        self.out_size = tuple(out_size)
        self.fallback_to_original = fallback_to_original

    def _to_rgb_for_detector(self, img: np.ndarray) -> np.ndarray:
        """Ritorna immagine RGB adatta a MTCNN. Accetta RGB o BGR in input."""
        img = np.asarray(img)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("L'immagine deve essere HxWx3")

        # euristica semplice per distinguere BGR da RGB: confronto medie canali
        b_mean = img[:, :, 0].mean()
        r_mean = img[:, :, 2].mean()
        if b_mean > r_mean:
            # probabilmente BGR
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # probabilmente già RGB
            return img.copy()

    def get_face_keypoints_mtcnn(self, image_bgr: np.ndarray):
        """Ritorna un array (5,2) con i punti (x,y) in float32 o None se non trova volti.

        Mantiene la firma del file originale (accetta immagini BGR), ma funziona anche con RGB.
        """
        if image_bgr is None:
            return None

        image_rgb = self._to_rgb_for_detector(image_bgr)

        boxes, probs, landmarks = self.mtcnn.detect(image_rgb, landmarks=True)
        # landmarks: array (n,5,2) in formato (x,y)
        if landmarks is None or len(landmarks) == 0:
            return None

        pts = landmarks[0].astype(np.float32)
        return pts

    def compute_affine(self, src_pts: np.ndarray, dst_pts: np.ndarray):
        """Calcola la trasformazione affine (estimateAffinePartial2D) e la ritorna."""
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
        """Allinea `img_src` a una posizione canonica e ritorna immagine RGB out_size x out_size.

        - Se non trova un volto e fallback_to_original=True, ritorna l'immagine ridimensionata a `out_size`.
        - Ideale per l'uso come transform `__call__`.
        """
        if img_src is None:
            return None

        out_size = out_size or self.out_size
        kp_src = self.get_face_keypoints_mtcnn(img_src)
        if kp_src is None:
            if self.fallback_to_original:
                resized = cv2.resize(img_src, (out_size[0], out_size[1]), interpolation=cv2.INTER_LINEAR)
                # Assicuriamoci che sia RGB
                rgb = self._to_rgb_for_detector(resized)
                return rgb
            else:
                raise RuntimeError("Nessun volto trovato nell'immagine")

        dst = self._canonical_dst(out_size)
        M = self.compute_affine(kp_src, dst)
        if M is None:
            if self.fallback_to_original:
                resized = cv2.resize(img_src, (out_size[0], out_size[1]), interpolation=cv2.INTER_LINEAR)
                rgb = self._to_rgb_for_detector(resized)
                return rgb
            else:
                raise RuntimeError("Impossibile calcolare la trasformazione affine per l'allineamento")

        # applichiamo la warp su immagine in formato RGB (per compatibilità con transforms)
        # se l'input era BGR, _to_rgb_for_detector lo ha convertito correttamente
        img_rgb = self._to_rgb_for_detector(img_src)
        aligned = cv2.warpAffine(img_rgb, M, (out_size[0], out_size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return aligned

    def __call__(self, img):
        """Convenienza: rende l'oggetto utilizzabile come Transform.

        Restituisce immagine RGB `out_size` pronta per essere passata a `transforms.ToPILImage()`.
        """
        return self.align_to_template(img)

# se runno diretto esegui un test semplice
if __name__ == "__main__":
    aligner = AlignerMtcnn()

    img_src_path = "Dataset/image0000416.jpg"

    img_src = cv2.imread(img_src_path)

    if img_src is None:
        raise RuntimeError(f"img_src = None (path sbagliato o file non leggibile): {img_src_path}")

    aligned = aligner.align_to_template(img_src, out_size=(224, 224))

    cv2.imshow("Original Source", img_src)
    cv2.imshow("Aligned Source", aligned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

