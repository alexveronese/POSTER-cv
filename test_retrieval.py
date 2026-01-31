import torch
import faiss
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from models.emotion_hyp import pyramid_trans_expr
from utils import *
import os


def test_inference():
    # --- CONFIGURAZIONE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './POSTER/checkpoint/epoch262_acc0.9179.pth'
    index_path = "raf_db_memory.index"
    paths_path = "raf_db_memory_paths.npy"
    labels_path = "raf_db_memory_labels.npy"

    # Mapping delle emozioni per RAF-DB
    emotion_dict = {0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happy",
                    4: "Sad", 5: "Angry", 6: "Neutral"}

    # 1. CARICAMENTO MODELLO E INDICE
    print("⏳ Caricamento risorse...")
    model = pyramid_trans_expr(img_size=224, num_classes=7)
    checkpoint = torch.load(model_path, map_location=device)

    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)
    model.to(device)
    model.eval()

    index = faiss.read_index(index_path)
    # Se hai una GPU, puoi spostare l'indice qui per ricerche fulminee
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    db_paths = np.load(paths_path)
    db_labels = np.load(labels_path)

    # 2. PREPARAZIONE IMMAGINE DI TEST (QUERY)
    # Scegli un'immagine dal tuo test set
    query_img_path = "./POSTER/saved_very_sad.jpg"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_bgr = cv2.imread(query_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # 3. ESTRAZIONE E RETRIEVAL
    with torch.no_grad():
        logits, y_feat = model(img_tensor)
        query_feat = y_feat.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_feat)  # Fondamentale!

        # Cerchiamo i 5 vicini più simili
        k = 5
        distances, indices = index.search(query_feat, k)

    # 4. VISUALIZZAZIONE
    print(f"Ricerca completata. Visualizzazione in corso...")
    plt.figure(figsize=(20, 5))

    # Mostra la Query
    plt.subplot(1, k + 1, 1)
    plt.imshow(img_rgb)
    pred_label = torch.argmax(logits, dim=1).item()
    plt.title(f"QUERY (Input)\nPred: {emotion_dict[pred_label]}")
    plt.axis('off')

    # Mostra i Vicini
    for i in range(k):
        idx = indices[0][i]
        neighbor_img = cv2.cvtColor(cv2.imread(db_paths[idx]), cv2.COLOR_BGR2RGB)
        label_id = db_labels[idx]
        dist = distances[0][i]

        plt.subplot(1, k + 1, i + 2)
        plt.imshow(neighbor_img)
        plt.title(f"Vicino {i + 1}\nLabel: {emotion_dict[label_id]}\nSim: {dist:.3f}")
        plt.axis('off')

    output_dir = "risultati_retrieval"
    os.makedirs(output_dir, exist_ok=True)

    # Salva il grafico completo
    save_path = os.path.join(output_dir, "confronto_completo.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Grafico salvato in: {save_path}")
    # ========================================================

    plt.tight_layout()

if __name__ == "__main__":
    test_inference()