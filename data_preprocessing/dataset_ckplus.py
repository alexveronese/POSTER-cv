import cv2
import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image
import random

class CKplusDataSet(data.Dataset):

    ID_TO_EMOTION = {
        0: "Anger",
        1: "Disgust",
        2: "Fear",
        3: "Happiness",
        4: "Sadness",
        5: "Surprise",
        6: "Neutral",
        7: "Contempt"
    }

    def __init__(self, csv_file, train=True, transform=None, basic_aug=False, dataidxs=None):
        self.train = train
        self.transform = transform
        self.basic_aug = basic_aug
        self.dataidxs = dataidxs

        # Load CSV with pandas
        self.data_frame = pd.read_csv(csv_file)

        if self.dataidxs is not None:
            self.data_frame = self.data_frame.iloc[self.dataidxs].reset_index(drop=True)

        self.target = self.data_frame["emotion"].astype(int).tolist()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        label = int(row['emotion'])
        pixels_str = row['pixels']
        pixels = np.array(pixels_str.split(), dtype=np.uint8).reshape(48, 48)

        # Duplicate the single channel in 3 channels for consistency
        img = np.stack([pixels] * 3, axis=-1)  # (48, 48, 3)

        if self.train and self.basic_aug:
            if random.random() > 0.5:
                img = np.flip(img, axis=1).copy()

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def add_gaussian_noise(image_array, mean=0.0, var=30):
        std = var ** 0.5
        noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
        noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img_clipped

    @staticmethod
    def flip_image(image_array):
        return cv2.flip(image_array, 1)
