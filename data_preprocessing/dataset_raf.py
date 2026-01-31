import torch.utils.data as data
import cv2
import pandas as pd
import os
import random
import numpy as np

class DataSetLoader(data.Dataset):
    # Mapping from folder names (1-7/8) to emotion labels as strings
    # The directories must be named as
    ID_TO_EMOTION = {
        1: "Surprise",
        2: "Fear",
        3: "Disgust",
        4: "Happy",
        5: "Sad",
        6: "Angry",
        7: "Neutral",
        8: "Contempt"
    }

    def __init__(self, datadir, train=True, transform=None, basic_aug=False, dataidxs=None):
        """
        Dataset loader for RAF database images organized by folder for each emotion class.

        Args:
            datadir (str): Root directory of RAF dataset.
            train (bool): Whether to load training set or test set.
            transform (callable or None): Optional transform function to apply on images.
            basic_aug (bool): Whether to enable simple augmentations (flip, noise).
            dataidxs (list or None): Optional subset of indices for partial dataset loading.
        """
        self.train = train
        self.dataidxs = dataidxs
        self.transform = transform
        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]  # List of basic augmentation functions

        # Determine whether to load from 'train' or 'test' subfolder
        split_dir = "train" if self.train else "test"
        data_dir = os.path.join(datadir, split_dir)

        # Check if the directory exists, raise error if not
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Folder {data_dir} not found!")

        self.file_paths = []  # Store full paths to image files
        self.target = []  # Store corresponding labels

        # Scan through the folders inside the chosen split directory (expect folders named '1', '2', ... '7')
        for folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder)

            # Skip non-directory files (e.g., hidden files)
            if not os.path.isdir(folder_path):
                continue

            try:
                label = int(folder) - 1  # Convert folder name '1'–'7' to label 0–6
            except ValueError:
                print(f"Ignored non-numeric folder: {folder}")
                continue

            # Iterate all images in folder and record their paths and labels
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.file_paths.append(os.path.join(folder_path, fname))
                    self.target.append(label)

        # Convert lists to numpy arrays for convenient indexing and slicing
        self.file_paths = np.array(self.file_paths)
        self.target = np.array(self.target)

        # If a subset of indices (`dataidxs`) is provided, select only those samples
        if self.dataidxs is not None:
            self.file_paths = self.file_paths[self.dataidxs]
            self.target = self.target[self.dataidxs]

        self.file_paths = self.file_paths.tolist()

        print(f"Loaded {len(self.file_paths)} images from {split_dir}/ ({len(set(self.target))} classes)")

    def __len__(self):
        # Return number of samples in dataset
        return len(self.file_paths)

    def get_labels(self):
        # Return array of target labels
        return self.target

    def __getitem__(self, idx):
        # Load image and corresponding label at index 'idx'
        path = self.file_paths[idx]
        img = cv2.imread(path)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV format) to RGB

        target = self.target[idx]

        # Apply basic augmentation randomly if enabled and in train mode
        if self.train:
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                img = self.aug_func[index](img)

        # If additional transform pipeline is supplied, apply it to image
        if self.transform is not None:
            img = self.transform(img)

        # Return augmented/transformed image and label tuple
        return img, target, path  # optionally can return idx as well


def add_gaussian_noise(image_array, mean=0.0, var=30):
    # Add Gaussian noise to the image array and clip pixel values to valid range [0, 255]
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped


def flip_image(image_array):
    # Flip the image horizontally
    return cv2.flip(image_array, 1)
