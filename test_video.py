import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch
import os
import argparse
from data_preprocessing.dataset_raf import DataSetLoader
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class

from utils import *
from models.emotion_hyp import pyramid_trans_expr
from sklearn.metrics import confusion_matrix
from data_preprocessing.plot_confusion_matrix import plot_confusion_matrix
from Aligment.Aligment import AlignerMtcnn
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm.notebook import tqdm
from facenet_pytorch import (MTCNN)
from PIL import Image
import cv2



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--workers', default=2, type=int, help='Number of dataset loading workers (default: 4)')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    parser.add_argument('-p', '--plot_cm', action="store_true", help="Ploting confusion matrix.")
    return parser.parse_args()

def test():
    print("start test")
    args = parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    #print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Using device:", device)

    aligner = AlignerMtcnn(device='cpu', out_size=(224, 224))

    # Initialize MTCNN model for single face cropping
    mtcnn = MTCNN(
        image_size=224,
        margin=0,
        min_face_size=200,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        keep_all=False,
        device=device
    )

    data_transforms_test = transforms.Compose([
        aligner,
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    num_classes = 7
    ID_TO_EMOTION = {
        0: "Surprise",
        1: "Fear",
        2: "Disgust",
        3: "Happy",
        4: "Sad",
        5: "Angry",
        6: "Neutral",
    }


    # Load your video
    scene = "C:\\Users\\veron\\Downloads\\lalaland.mov"
    clip = VideoFileClip(scene)
    # Save video frames per second
    vid_fps = clip.fps
    # Get the video (as frames)
    video = clip.without_audio()
    video_data = np.array(list(video.iter_frames()))
    #print(vid_fps)

    skips = 120
    reduced_video = []


    for i in tqdm(range(0, len(video_data), skips)):
        temporary = Image.fromarray(video_data[i]).copy()
        sample = mtcnn.detect(temporary)
        if sample[0] is not None:
            box = sample[0][0]
            face = temporary.crop(box)
            reduced_video.append(face)

    #reduced_video[0].show()
    print(len(reduced_video))
    """
    path = "C:\\Users\\veron\\Downloads\\affectnet\\archive(3)\\Test\\happy\\ffhq_863.png"
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_dataset = data_transforms_test(img).unsqueeze(0).to(device)
    """

    model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    print("Loading pretrained weights...", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)

    for img in reduced_video:
        test_dataset = data_transforms_test(img).unsqueeze(0).to(device)
        test_size = test_dataset.__len__()
        print('Test set size:', test_size)

        model = model.to(device)

        model.eval()
        with torch.no_grad():
            labels, features = model(test_dataset)
            _, predicts = torch.max(labels, 1)

        print(labels)
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow(str(ID_TO_EMOTION[predicts.numpy()[0]]), opencvImage)
        print(ID_TO_EMOTION[predicts.numpy()[0]])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    pre_labels = []
    gt_labels = []
    with torch.no_grad():
        bingo_cnt = 0
        model.eval()
        for batch_i, (imgs, targets) in enumerate(test_loader):
            outputs, features = model(imgs.cuda())
            targets = targets.cuda()
            _, predicts = torch.max(outputs, 1)
            _, predicts = torch.max(outputs, 1)
            correct_or_not = torch.eq(predicts, targets)
            bingo_cnt += correct_or_not.sum().cpu()
            pre_labels += predicts.cpu().tolist()
            gt_labels += targets.cpu().tolist()
   
    
        acc = bingo_cnt.float() / float(test_size)
        acc = np.around(acc.numpy(), 4)
        print(f"Test accuracy: {acc:.4f}.")
        cm = confusion_matrix(gt_labels, pre_labels)
        # print(cm)

    if args.plot_cm:
        cm = confusion_matrix(gt_labels, pre_labels)
        cm = np.array(cm)
        if args.dataset == "rafdb":
            labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]  #
            plot_confusion_matrix(cm, labels_name, 'RAF-DB', acc)

        if args.dataset == "affectnet":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN"]  #
            plot_confusion_matrix(cm, labels_name, 'AffectNet', acc)

        if args.dataset == "affectnet8class":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN", "CO"]  #
            # 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,
            # 7: Contempt,
            plot_confusion_matrix(cm, labels_name, 'AffectNet_8class', acc)
    """



if __name__ == "__main__":                    
    test()

