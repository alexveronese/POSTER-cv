from turtle import mode
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
import matplotlib.pyplot as plt
from utils import *
from models.emotion_hyp import pyramid_trans_expr
from Aligment.Aligment import AlignerMtcnn
#from moviepy.editor import VideoFileClip, ImageSequenceClip
#from tqdm.notebook import tqdm
#from facenet_pytorch import (MTCNN)
from PIL import Image
import cv2
import seaborn as sns



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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    aligner = AlignerMtcnn(device='cpu', out_size=(224, 224))

    # Initialize MTCNN model for single face cropping
    """
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
"""
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
    
    model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    print("Loading pretrained weights...", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)
    # Load your video
    scene = "/Users/Bea_1/Desktop/lalaland_cut.mov"
    
    """
    clip = VideoFileClip(scene) 
    # Save video frames per second
    vid_fps = clip.fps
    video = clip.without_audio()
    video_data = np.array(list(video.iter_frames()))
    cv2_video = cv2.VideoCapture(scene)
    # Get the video (as frames)
    #print(vid_fps)
    """
    
    cv2_video = cv2.VideoCapture(scene)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    colors = {
            "Angry": "red",
            "Disgust": "green",
            "Fear": "gray",
            "Happy": "yellow",
            "Neutral": "purple",
            "Sad": "blue",
            "Surprise": "orange"
        }
    model.eval()
    i = 0
    status = ""
    plt.ion()  # interacting modality
    fig, axs = plt.subplots(1, 1, figsize=(4, 2))
    window_name = "Emotion video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)

    manager = plt.get_current_fig_manager()
    try:
        # Typic backend  (Qt5Agg)
        manager.window.move(0, 800)   # (x=0, y=800)
    except Exception:
        try:
            # Backend TkAgg
            manager.window.wm_geometry("+0+800")
        except Exception:
            pass
    
    while cv2_video.isOpened():
        ret, frame = cv2_video.read()
        
        if not ret:
            break
        
        #if (i == 0 or i == 30):
        i = 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0: 
            continue
        
        x, y, w, h = faces[0]
        face_img_orig = frame[y:y+h, x:x+w]
        face_img = cv2.cvtColor(face_img_orig, cv2.COLOR_BGR2RGB)
        
    
        test_dataset = data_transforms_test(face_img).unsqueeze(0).to(device)
        with torch.no_grad():
            labels, features = model(test_dataset)
            _, predicts = torch.max(labels, 1)

            probabilities = torch.nn.functional.softmax(labels, dim=-1)
            probabilities = probabilities.detach().numpy().tolist()[0]

            if probabilities[3] >=0.5:
                print(str(ID_TO_EMOTION[predicts.numpy()[0]]) + "save image")
                cv2.imwrite("saved_happy1.jpg", face_img_orig)

            class_probabilities = {ID_TO_EMOTION[i] : prob for i,
                               prob in enumerate(probabilities)}
            palette = [colors[label] for label in class_probabilities.keys()]
            axs.clear()
            sns.barplot(ax=axs,
                    y=list(class_probabilities.keys()),
                    x=[prob * 100 for prob in class_probabilities.values()],
                    palette=palette,
                    orient='h')
            axs.set_xlabel('Probability (%)')
            axs.set_title('Emotion Probabilities')
            axs.set_xlim([0, 100])

            plt.tight_layout()
            plt.draw()
            plt.pause(0.1) 
            cv2.putText(frame, ID_TO_EMOTION[predicts.numpy()[0]], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            status = ID_TO_EMOTION[predicts.numpy()[0]]
            
        #i += 1
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow(window_name, frame)
        cv2.moveWindow(window_name, 0, 0, )
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2_video.release()



    """
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
    """
    path = "C:\\Users\\veron\\Downloads\\affectnet\\archive(3)\\Test\\happy\\ffhq_863.png"
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_dataset = data_transforms_test(img).unsqueeze(0).to(device)
    



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
    """
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

