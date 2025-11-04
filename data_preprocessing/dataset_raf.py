import torch.utils.data as data
import cv2
import pandas as pd
import os
# import image_utils
import random
import cv2
import numpy as np
from torchvision import datasets


class RafDataSet(data.Dataset):

    ID_TO_EMOTION = {
        1: "Surprise",
        2: "Fear",
        3: "Disgust",
        4: "Happy",
        5: "Sad",
        6: "Angry",
        7: "Neutral"
    }

    def __init__(self, raf_path, dataidxs=None, train=True, transform=None, basic_aug=False):
        self.train = train
        self.dataidxs = dataidxs
        self.transform = transform
        self.raf_path = raf_path
        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]

        # === Seleziona split (train/test) ===
        split_dir = "train" if self.train else "test"
        data_dir = os.path.join(self.raf_path, split_dir)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Cartella {data_dir} non trovata!")

        self.file_paths = []
        self.target = []

        # === Scansiona le cartelle numeriche (1-7) ===
        for folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                label = int(folder) - 1  # converte 1–7 → 0–6
            except ValueError:
                print(f"Cartella ignorata (non numerica): {folder}")
                continue

            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.file_paths.append(os.path.join(folder_path, fname))
                    self.target.append(label)

        self.file_paths = np.array(self.file_paths)
        self.target = np.array(self.target)

        # === Gestione subset opzionale ===
        if self.dataidxs is not None:
            self.file_paths = self.file_paths[self.dataidxs]
            self.target = self.target[self.dataidxs]
        self.file_paths = self.file_paths.tolist()

        print(f"Caricate {len(self.file_paths)} immagini da {split_dir}/ ({len(set(self.target))} classi)")


    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.target

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        sample = cv2.imread(path)
        # sample = sample[:, :, ::-1]  # BGR to RGB
        target = self.target[idx]
        if self.train:
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                sample = self.aug_func[index](sample)

        if self.transform is not None:
            sample = self.transform(sample.copy())

        return sample, target #, idx


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

