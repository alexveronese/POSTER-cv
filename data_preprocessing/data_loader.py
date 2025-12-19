import logging

import numpy as np
from numpy.core.fromnumeric import mean
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data_preprocessing.dataset_ferplus import FerPlusDataSet
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_ckplus import CKplusDataSet

from PIL import Image
from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision

from torch.utils.data.sampler import Sampler

# Configure logging with INFO level for displaying informational messages
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def record_net_data_stats(y_train, net_dataidx_map):
    """
    Calculate and record the number of samples per class on each client.

    Args:
        y_train (np.array): Array of training labels.
        net_dataidx_map (dict): Dictionary mapping client id to dataset indices.

    Returns:
        dict: Dictionary with client id as key and class count dict as value.
    """
    net_cls_counts = {}

    # Iterate through clients and count classes in their dataset partitions
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    # Debug log for dataset distribution (disabled by default, can be enabled if needed)
    logging.debug('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


class Lighting(object):
    """
    Implements PCA-based lighting noise augmentation as described in AlexNet paper.

    This augmentation perturbs the colors of the image according to principal components
    of the ImageNet dataset, adding subtle but useful lighting variation for robustness.
    """
    imagenet_pca = {
        'eigval': np.asarray([0.2175, 0.0188, 0.0045]),  # Eigenvalues for principal components
        'eigvec': np.asarray([                           # Eigenvectors for principal components
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        """
        Apply the lighting noise augmentation to an image.

        Args:
            img (PIL Image or numpy.array): Input image.

        Returns:
            PIL Image: Augmented image.
        """
        if self.alphastd == 0.:
            return img  # No augmentation if stddev is zero

        rnd = np.random.randn(3) * self.alphastd  # Random noise vector
        rnd = rnd.astype('float32')
        v = rnd * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))

        old_dtype = np.asarray(img).dtype
        img = np.asarray(img).astype('float32')
        img = img + inc  # Add perturbation to image pixels

        # Clip image to valid uint8 range and convert back to original type if necessary
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)

        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.target
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples



def _data_transforms_raf(datadir, use_lighting=False):
    """
    Define the dataset augmentation and normalization transforms for RAF dataset.

    Returns:
        train_transform (Compose): Transformations applied on training samples.
        valid_transform (Compose): Transformations applied on validation/test samples.
    """
    train_transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224))
    ]

    if use_lighting:
        train_transform_list.append(Lighting(0.1))  # Lighting augmentation

    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1))
    ])

    train_transform = transforms.Compose(train_transform_list)

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, valid_transform

def _data_transforms_affectnet(datadir, use_lighting=False):
    train_transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224))
    ]

    if use_lighting:
        train_transform_list.append(Lighting(0.1))  # Lighting augmentation

    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1))
    ])

    train_transform = transforms.Compose(train_transform_list)

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, valid_transform

def _data_transforms_ckplus(datadir, use_lighting=False):
    train_transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224))
    ]

    if use_lighting:
        train_transform_list.append(Lighting(0.1))  # Lighting augmentation

    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(scale=(0.02, 0.1))
    ])

    train_transform = transforms.Compose(train_transform_list)

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, valid_transform


def load_data(datadir):
    """
    Load dataset and apply appropriate transformations depending on dataset type.

    Args:
        datadir (str): Path to dataset directory.

    Returns:
        Tuple[np.array, np.array]: Training and test labels arrays.
    """
    if "raf" in datadir:
        train_transform, test_transform = _data_transforms_raf(datadir)
        dl_obj = RafDataSet
        train_ds = dl_obj(datadir, train=True, transform=train_transform, basic_aug=True)
        test_ds = dl_obj(datadir, train=False, transform=test_transform)
        y_train, y_test = train_ds.target, test_ds.target  # Assumes .target attributes contain labels
    elif "AffectNet" in datadir:
        train_transform, test_transform = _data_transforms_affectnet(datadir)
        dl_obj = Affectdataset
        train_ds = dl_obj(datadir, train=True, transform=train_transform, basic_aug=True)
        test_ds = dl_obj(datadir, train=False, transform=test_transform)
        # y_train, y_test = train_ds.label, test_ds.label
        y_train, y_test = train_ds.target, test_ds.target
    elif "Ckplus" in datadir:
        train_transform, test_transform = _data_transforms_ckplus(datadir)
        dl_obj = CKplusDataSet
        train_ds = dl_obj(datadir, train=True, transform=train_transform, basic_aug=True)
        test_ds = dl_obj(datadir, train=False, transform=test_transform)
        y_train, y_test = train_ds.target, test_ds.target
    elif "FerPlus" in datadir:
        train_transform, test_transform = _data_transforms_ckplus(datadir)
        dl_obj = FerPlusDataSet
        train_ds = dl_obj(datadir, train=True, transform=train_transform, basic_aug=True)
        test_ds = dl_obj(datadir, train=False, transform=test_transform)
        y_train, y_test = train_ds.target, test_ds.target


    return (y_train, y_test)


def partition_data(datadir, partition, n_nets, alpha):
    """
    Partition dataset indices among clients for federated learning training.

    Args:
        datadir (str): Dataset path.
        partition (str): 'homo' for homogeneous(IID), 'hetero' for heterogeneous(non-IID).
        n_nets (int): Number of clients.
        alpha (float): Dirichlet distribution parameter (only for hetero).

    Returns:
        class_num (int): Number of classes.
        net_dataidx_map (dict): Mapping client_id -> dataset indices list.
        traindata_cls_counts (dict): Class counts per client.
    """
    logging.info("*********partition dataset***************")
    y_train, y_test = load_data(datadir)
    n_train = y_train.shape[0]
    class_num = len(np.unique(y_train))

    if partition == "homo":
        # IID partitioning: shuffle and split equally
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        # non-IID partitioning using Dirichlet distribution to simulate class imbalance across clients
        min_size = 0
        K = class_num
        N = n_train
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                # Balance to limit size of partitions
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                # Split indices according to proportions
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # Record class counts per client dataset partition
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return class_num, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(datadir, train_bs, test_bs, balanced_sampler, dataidxs=None):
    """
       Prepare PyTorch DataLoaders for training and testing datasets.

       Args:
           datadir (str): Dataset path.
           train_bs (int): Training batch size.
           test_bs (int): Test batch size.
           balanced_sampler (bool): Whether to use balanced sampler for minority classes.
           dataidxs (list, optional): Subset of indices to load. Defaults to None.

       Returns:
           Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """

    if 'raf' in datadir:
        train_transform, test_transform = _data_transforms_raf(datadir)
        dl_obj = RafDataSet
        workers = 4
        persist = False

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=train_transform, download=True)
    test_ds = dl_obj(datadir, train=False, transform=test_transform, download=True)
    if balanced_sampler:
        sampler = ImbalancedDatasetSampler(train_ds)
    else:
        sampler = None
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=(sampler is None), drop_last=True,
                               num_workers=workers, persistent_workers=persist, sampler=sampler)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=workers,
                              persistent_workers=persist)

    return train_dl, test_dl


def load_partition_data(data_dir, partition_method, partition_alpha, client_number, batch_size, balanced_sampler):
    """
    Load dataset partitions and prepare DataLoaders for federated learning.

    Args:
        data_dir (str): Dataset directory path.
        partition_method (str): 'homo' or 'hetero' partitioning method.
        partition_alpha (float): Alpha parameter controlling heterogeneity.
        client_number (int): Number of clients.
        batch_size (int): Batch size for loaders.
        balanced_sampler (bool): Whether to use balanced sampler.

    Returns:
        tuple: Contains overall train/test sample counts, global loaders,
               local sample counts, local loaders dicts, and number of classes.
    """
    class_num, net_dataidx_map, traindata_cls_counts = partition_data(data_dir, partition_method, client_number, partition_alpha)

    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # Load global train/test dataloaders with all dataset
    train_data_global, test_data_global = get_dataloader(data_dir, batch_size, batch_size, balanced_sampler)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # Prepare local datasets and loaders for each client
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        train_data_local, test_data_local = get_dataloader(data_dir, batch_size, batch_size,
                                                          balanced_sampler, dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" %
                     (client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num