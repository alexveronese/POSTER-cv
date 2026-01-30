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
    # Assumi di caricare un'immagine
    IMAGE_PATH = "C:\\Users\\veron\\Desktop\\Uni\\LM\\1_ANNO\\2_SEM\\Computer_Vision\\POSTER-cv\\saved_sad_top.jpg"

    aligner = AlignerMtcnn(device='cpu', out_size=(224, 224))

    # 1. Preparazione dell'Immagine
    data_transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    original_image_pil = Image.open(IMAGE_PATH).convert('RGB')
    aligned_face_pil = aligner(original_image_pil)
    aligned_face_np = np.array(aligned_face_pil)
    #original_image_np = np.array(original_image_pil) # [H, W, C]
    input_tensor = data_transforms_test(aligned_face_pil).unsqueeze(0).to(device) # [1, C, H, W]

    # 2. Caricamento del Modello
    model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type="large") # usa il tuo args.modeltype
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)

    # Se hai salvato un modello LoRA avvolto:
    # Devi prima avvolgere l'istanza di pyramid_fuse con LoRAConfig vuota
    # e poi chiamare model.load_state_dict(state_dict)

    """
    model.eval()
    labels, _ = model(input_tensor)
    _, predicts = torch.max(labels, 1)
    print(labels)
    print(predicts.numpy()[0])
    """

    # 3. Inizializza Grad-CAM
    # Usa 'ir_back.layer4' per il modello non avvolto da DataParallel
    TARGET_LAYER = 'ir_back.body.20'
    grad_cam_instance = GradCAM(model, target_layer_name=TARGET_LAYER)

    # 4. Calcola e Visualizza
    cam_map, predicted_class = grad_cam_instance(input_tensor)

    # L'immagine originale deve essere ridimensionata per l'overlay se la CAM è 224x224
    resized_image = cv2.resize(aligned_face_np, (224, 224))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    visualized_image = overlay_cam(resized_image, cam_map)

    # Converto da RGB a BGR per OpenCV (se necessario) e salvo
    cv2.imwrite("grad_sad_top.jpg", visualized_image)
    print(f"Predizione: Classe {predicted_class}. Immagine salvata come grad_cam_result.jpg")



if __name__ == "__main__":
    cam()