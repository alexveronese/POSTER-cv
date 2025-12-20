import torch
import torchvision.transforms as transforms
from models.emotion_hyp import pyramid_trans_expr
from utils import GradCAM, overlay_cam
from PIL import Image
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
# Assumi di caricare un'immagine
IMAGE_PATH = 'path/to/your/test/image.jpg'
CHECKPOINT_PATH = 'path/to/your/checkpoint.pth'

# 1. Preparazione dell'Immagine
data_transforms_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

original_image_pil = Image.open(IMAGE_PATH).convert('RGB')
original_image_np = np.array(original_image_pil) # [H, W, C]
input_tensor = data_transforms_val(original_image_pil).unsqueeze(0).to(device) # [1, C, H, W]

# 2. Caricamento del Modello
model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type="large") # usa il tuo args.modeltype
model.to(device)

# Carica i pesi: Devi sapere se il checkpoint è l'intero modello o solo LoRA
checkpoint = torch.load(CHECKPOINT_PATH)
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

# Se hai salvato l'intero modello o un modello solo-LoRA con state_dict intero:
model.load_state_dict(state_dict)

# Se hai salvato un modello LoRA avvolto:
# Devi prima avvolgere l'istanza di pyramid_fuse con LoRAConfig vuota
# e poi chiamare model.load_state_dict(state_dict)

model.eval()

# 3. Inizializza Grad-CAM
# Usa 'ir_back.layer4' per il modello non avvolto da DataParallel
TARGET_LAYER = 'ir_back.body.20'
grad_cam_instance = GradCAM(model, target_layer_name=TARGET_LAYER)

# 4. Calcola e Visualizza
cam_map, predicted_class = grad_cam_instance(input_tensor)

# L'immagine originale deve essere ridimensionata per l'overlay se la CAM è 224x224
resized_image = cv2.resize(original_image_np, (224, 224))
visualized_image = overlay_cam(resized_image, cam_map)

# Converto da RGB a BGR per OpenCV (se necessario) e salvo
cv2.imwrite("grad_cam_result.jpg", cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
print(f"Predizione: Classe {predicted_class}. Immagine salvata come grad_cam_result.jpg")