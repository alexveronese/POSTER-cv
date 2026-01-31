import os
import torch
import numpy as np
import faiss
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import *
from data_preprocessing.dataset_raf import DataSetLoader
from models.emotion_hyp import pyramid_trans_expr


def build():
    # --- CONFIGURAZIONE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = './POSTER/dataset/RafDataSet'
    model_path = './POSTER/checkpoint/epoch262_acc0.9179.pth'
    save_name = "raf_db_memory"
    batch_size = 32
    img_size = 224

    print(f"Using device: {device}")

    # 1. TRASFORMAZIONI (Usa le stesse del tuo addestramento)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. DATALOADER
    # Nota: Assicurati di aver modificato __getitem__ per restituire (img, target, path)
    train_dataset = DataSetLoader(data_dir, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3. CARICAMENTO MODELLO
    print("Loading POSTER model...")
    model = pyramid_trans_expr(img_size=img_size, num_classes=7)
    checkpoint = torch.load(model_path, map_location=device)

    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)

    model.to(device)
    model.eval()

    # 4. ESTRAZIONE FEATURE
    all_features = []
    all_paths = []
    all_labels = []

    print(f"Extracting features for {len(train_dataset)} images...")
    with torch.no_grad():
        for imgs, targets, paths in tqdm(train_loader):
            imgs = imgs.to(device)

            # Il tuo modello restituisce (out, y_feat)
            _, y_feat = model(imgs)

            # Normalizzazione e conversione
            feat = y_feat.cpu().numpy().astype('float32')
            all_features.append(feat)

            all_paths.extend(paths)
            all_labels.extend(targets.numpy())

    # Concatenazione
    all_features = np.vstack(all_features)

    # ... (parti 1, 2, 3, 4 rimangono uguali) ...

    # 5. CREAZIONE INDICE FAISS (Versione GPU)
    print("Building FAISS index on GPU...")
                
    # Normalizziamo i vettori (importante per la somiglianza del coseno)
    faiss.normalize_L2(all_features)
              
    d = all_features.shape[1] # dimensione 512
                                   
    # Passaggio A: Creiamo l'indice base su CPU
    cpu_index = faiss.IndexFlatIP(d)
                                               
    # Passaggio B: Lo trasferiamo sulla GPU
    res = faiss.StandardGpuResources() # Inizializza le risorse GPU per FAISS
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index) # 0 è l'ID della tua scheda video
                                                                
    # Passaggio C: Aggiungiamo i vettori estratti (operazione velocissima su GPU)
    gpu_index.add(all_features)

    # 6. SALVATAGGIO
    # FAISS non può salvare direttamente un indice GPU su disco.
    # Dobbiamo prima riportarlo su CPU (solo per il salvataggio).
    index_to_save = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(index_to_save, f"{save_name}.index")
                                                                                               
    # Salviamo i metadati come prima
    np.save(f"{save_name}_paths.npy", np.array(all_paths))
    np.save(f"{save_name}_labels.npy", np.array(all_labels))


if __name__ == "__main__":
    build()
