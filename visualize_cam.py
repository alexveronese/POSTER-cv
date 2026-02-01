import torch
import torchvision.transforms as transforms

from Aligment.Aligment import AlignerMtcnn
from models.emotion_hyp import pyramid_trans_expr
from utils import *
from PIL import Image
import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    return parser.parse_args()

def cam():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7

    IMAGE_PATH = "/Users/Bea_1/PycharmProjects/POSTER-cv/saved_happy1.jpg"

    aligner = AlignerMtcnn(device='cpu', out_size=(224, 224))


    data_transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    original_image_pil = Image.open(IMAGE_PATH).convert('RGB')

    aligned_face_pil = aligner(original_image_pil)
    aligned_face_np = np.array(aligned_face_pil)
    original_image_np = np.array(original_image_pil) # [H, W, C]
    input_tensor = data_transforms_test(aligned_face_pil).unsqueeze(0).to(device) # [1, C, H, W]
    #input_tensor = data_transforms_test(original_image_np).unsqueeze(0).to(device)  # [1, C, H, W]


    model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type="large") # usa il tuo args.modeltype
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)




    TARGET_LAYER = 'ir_back.body.20'
    grad_cam_instance = GradCAM(model, target_layer_name=TARGET_LAYER)


    cam_map, predicted_class = grad_cam_instance(input_tensor)


    resized_image = cv2.resize(aligned_face_np, (224, 224))
    # resized_image = cv2.resize(original_image_np, (224, 224))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    visualized_image = overlay_cam(resized_image, cam_map)


    cv2.imwrite("grad_aligned_happy1.jpg", visualized_image)
    print(f"Predizione: Classe {predicted_class}. Immagine salvata come grad_cam_result.jpg")



if __name__ == "__main__":
    cam()