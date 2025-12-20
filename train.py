import warnings

# Ignore warning messages to keep output clean
warnings.filterwarnings("ignore")

import numpy as np
import torch.utils.data as data
from torchvision import transforms
import os
import torch
import argparse
from collections import Counter

# Import the custom dataset class
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_ferplus import FerPlusDataSet
from data_preprocessing.dataset_ckplus import CKplusDataSet

# Import performance metrics from sklearn
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score

from time import time

# Import utility functions and custom optimizer (SAM), model architecture
from utils import *
from data_preprocessing.sam import SAM
from models.emotion_hyp import pyramid_trans_expr

# Import balanced sampler to address class imbalance during training
from data_preprocessing.data_loader import ImbalancedDatasetSampler

from data_preprocessing.data_loader import _data_transforms_raf
from data_preprocessing.data_loader import _data_transforms_affectnet
from data_preprocessing.data_loader import _data_transforms_ckplus


def parse_args():
    """
    Argument parser for command line parameters configuring dataset, model, batch size, optimizer, etc.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb', help='dataset name')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Path to PyTorch checkpoint file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--modeltype', type=str, default='large', help='Model size: small, base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Choose optimizer: adam or sgd')
    parser.add_argument('--lr', type=float, default=0.00004, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD optimizer')
    parser.add_argument('--workers', default=2, type=int, help='Number of dataset loader workers')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--gpu', type=str, default='0,1', help='Comma-separated list of GPUs to use')
    parser.add_argument('--loss', type=str, default='CE_loss', help='Loss function')    # alternative CB_loss
    parser.add_argument('--sampler', type=str, default='imbalanced', help='ImbalancedDatasetSampler')   # pass 'balanced' if you want to do the balance
    parser.add_argument('--lighting', action='store_true', help='Enable lighting augmentation')     # if '--lighting' is passed, args.lighting == True
    return parser.parse_args()


def run_training():
    """
    Runs the full training loop including dataset loading, model setup,
    optimizer, training and validation, as well as saving best checkpoints.
    """
    args = parse_args()
    torch.manual_seed(123)  # Set fixed seed for reproducibility

    # Select computation device: CUDA if available, else Apple MPS if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Using device:", device)

    # Define augmentation pipeline for training dataset
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),                 # Convert ndarray to PIL Image
        transforms.Resize((224, 224)),           # Resize to 224x224
        transforms.ToTensor(),                   # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])

    # Validation transforms - no augmentation except resize and normalization
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    num_classes = 7
    use_lighting = getattr(args, 'lighting', False)

    # Load dataset and model based on input arguments
    if args.dataset == "rafdb":
        datapath = './dataset/RafDataSet/'

        train_transform, valid_transform = _data_transforms_raf(datapath, use_lighting=use_lighting)

        # Initialize training and validation datasets
        train_dataset = RafDataSet(datapath, train=True, transform=train_transform, basic_aug=True)
        val_dataset = RafDataSet(datapath, train=False, transform=data_transforms_val)

        # Create the model with specified type and input config
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)
    elif args.dataset == "affectnet":
        datapath = './dataset/AffectNetDataSet/'

        train_transform, valid_transform = _data_transforms_affectnet(datapath, use_lighting=use_lighting)

        # Initialize training and validation datasets
        train_dataset = Affectdataset(datapath, train=True, transform=train_transform, basic_aug=True)
        val_dataset = Affectdataset(datapath, train=False, transform=data_transforms_val)

        # Create the model with specified type and input config
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)
    elif args.dataset == "ferplus":
        datapath = './dataset/FerPlusDataSet/'

        # Initialize training and validation datasets
        train_dataset = FerPlusDataSet(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = FerPlusDataSet(datapath, train=False, transform=data_transforms_val)

        # Create the model with specified type and input config
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)
    elif args.dataset == "ck+":
        datapath = './dataset/CK+DataSet/ckextended.csv'
        num_classes = 8

        train_transform, val_transform = _data_transforms_ckplus(datapath, use_lighting=use_lighting)

        # Split train and test data
        train_idx, val_idx = train_val_split(datapath, val_ratio=0.2)

        # Initialize training and validation datasets
        train_dataset = CKplusDataSet(datapath, train=True, transform=train_transform, basic_aug=True, dataidxs=train_idx)
        val_dataset = CKplusDataSet(datapath, train=False, transform=val_transform, basic_aug=False, dataidxs=val_idx)

        # Create the model with specified type and input config
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)
    else:
        # Handle unsupported dataset name
        return print('dataset name is not correct')

    val_num = val_dataset.__len__()

    # Print dataset sizes
    print('Train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())

    # Compute class distribution to use with Class Balanced Loss
    labels = train_dataset.target if hasattr(train_dataset, 'target') else [label for _, label in train_dataset]
    label_counts = Counter(labels)
    samples_per_cls = [label_counts[i] if i in label_counts else 0 for i in range(num_classes)]
    print(f"Samples per class for CB loss: {samples_per_cls}")


    if args.sampler == 'balanced':
        # Balanced sampler to address class imbalance in training set
        sampler = ImbalancedDatasetSampler(train_dataset)
    else:
        sampler = None

    # Create DataLoader for training with the balanced sampler and appropriate settings
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,                    # Use balanced sampler
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,                     # Do not shuffle when using sampler
        pin_memory=True)

    # Validation DataLoader without sampler and shuffle
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True)

    # Wrap model for multi-GPU dataset parallelism
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    print("batch_size:", args.batch_size)

    # Load pretrained weights if checkpoint specified
    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights(model, checkpoint)

    # Select optimizer based on argument
    params = model.parameters()
    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    # Use SAM optimizer for sharpness-aware training
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False)

    # Learning rate scheduler with exponential decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Calculate and print total number of trainable parameters in millions
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)

    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

    best_acc = 0

    # Early stopping parameters
    early_stopping_patience = 20
    epochs_no_improve = 0
    early_stopping_metric = 0

    # Training epochs loop
    for i in range(1, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        start_time = time()

        model.train()  # Set model to training mode

        # Training batch loop
        for batch_i, (imgs, targets) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            outputs, features = model(imgs)  # Forward pass to get predictions and features
            targets = targets.to(device)

            if args.loss == 'CB_loss':
                # Compute loss using CB_loss from utils.py
                loss = CB_loss(targets, outputs, samples_per_cls, num_classes, loss_type="focal", beta=0.9999, gamma=2.0)

                loss.backward()
                optimizer.first_step(zero_grad=True)  # SAM first step

                # Second forward-backward pass in SAM optimizer
                outputs, features = model(imgs)
                loss = CB_loss(targets, outputs, samples_per_cls, num_classes, loss_type="focal", beta=0.9999, gamma=2.0)

                loss.backward()  # full forward pass backward
                optimizer.second_step(zero_grad=True)  # SAM second step

            elif args.loss == 'CE_loss':
                CE_loss = CE_criterion(outputs, targets)
                lsce_loss = lsce_criterion(outputs, targets)
                loss = 2 * lsce_loss + CE_loss
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                outputs, features = model(imgs)
                CE_loss = CE_criterion(outputs, targets)
                lsce_loss = lsce_criterion(outputs, targets)

                loss = 2 * lsce_loss + CE_loss
                loss.backward()  # make sure to do a full forward pass
                optimizer.second_step(zero_grad=True)

            # Accumulate loss and accuracy stats
            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        # Calculate average training accuracy and loss
        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss / iter_cnt
        elapsed = (time() - start_time) / 60

        print('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Loss: %.3f LR:%.6f' %
              (i, elapsed, train_acc, train_loss, optimizer.param_groups[0]["lr"]))

        scheduler.step()  # Update learning rate

        # Validation loop with no gradient update
        pre_labels = []
        gt_labels = []
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()  # Set model to eval mode
            for batch_i, (imgs, targets) in enumerate(val_loader):
                outputs, features = model(imgs.to(device))
                targets = targets.to(device)

                if args.loss == 'CB_loss':
                    loss = CB_loss(targets, outputs, samples_per_cls, num_classes, loss_type="focal", beta=0.9999, gamma=2.0)
                elif args.loss == 'CE_loss':
                    loss = torch.nn.CrossEntropyLoss()(outputs, targets) # Use standard CE loss for validation

                val_loss += loss
                iter_cnt += 1

                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()

                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()

            # Compute validation metrics averaged over batches
            val_loss = val_loss / iter_cnt
            val_acc = bingo_cnt.float() / float(val_num)
            val_acc = np.around(val_acc.numpy(), 4)
            f1 = f1_score(pre_labels, gt_labels, average='macro')
            total_score = 0.67 * f1 + 0.33 * val_acc

            print("[Epoch %d] Validation accuracy:%.4f, Loss:%.3f, f1 %4f, score %4f" % (
                i, val_acc, val_loss, f1, total_score))

            # Save checkpoint if validation accuracy improves beyond threshold
            if val_acc > 0.907 and val_acc > best_acc:
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('./checkpoint', "epoch" + str(i) + "_acc" + str(val_acc) + ".pth"))
                print('Model saved.')

            if val_acc > best_acc:
                best_acc = val_acc
                print("best_acc:" + str(best_acc))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {i}. No improvement for {early_stopping_patience} epochs.")
                break


# Script entrypoint
if __name__ == "__main__":
    run_training()
