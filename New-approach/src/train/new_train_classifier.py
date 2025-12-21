import os, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.setup import new_config_cls as config 
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.cls_dataset_stream import GeometricShapeClassificationDatasetStream 
from src.models.resnet import build_resnet_classifier 
from src.utils.cls_viz import save_confusion_matrix, save_sample_grid
from src.dataio.split_utils import subsample_pairs
from src.utils.metrics_cls import summarize_classifier

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1)
            correct += (preds == y.to(device)).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. DATA LOAD
    train_p = subsample_pairs(paired_image_xml_list(config.IMG_DIR_TRAIN, config.XML_DIR_ALL), config.F_TRAIN)
    val_p   = subsample_pairs(paired_image_xml_list(config.IMG_DIR_VAL, config.XML_DIR_ALL), config.F_VAL)
    test_p  = subsample_pairs(paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL), config.F_TEST)

    # 2. LOADERS (Synthetic Augmentation only in train_loader)
    train_loader = DataLoader(GeometricShapeClassificationDatasetStream(train_p, train=True), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(GeometricShapeClassificationDatasetStream(val_p, train=False), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(GeometricShapeClassificationDatasetStream(test_p, train=False), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. INIT
    model = build_resnet_classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # 4. TRAINING LOOP
    best_val_acc = 0.0
    for ep in range(config.EPOCHS):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {ep+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        
        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        print(f"EP {ep+1} | Val_Acc (Squares): {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.SAVE_CKPT)

    # --- 5. FINAL VISUALIZATION ON RECTANGLES ---
    print("\n--- Final Test & Visualization (Rectangles) ---")
    model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
    model.eval()
    
    all_y, all_pred, sample_imgs = [], [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing Rectangles"):
            out = model(x.to(device))
            all_y.extend(y.tolist()); all_pred.extend(out.argmax(1).cpu().tolist())
            if len(sample_imgs) < 36: sample_imgs.extend(x.cpu())

    # Generate Confusion Matrix and Grid
    save_confusion_matrix(all_y, all_pred, out_path=os.path.join(config.OUTPUT_DIR, "confmat_area_final.png"), class_names=config.AREA_NAMES)
    save_sample_grid(torch.stack(sample_imgs[:36]), all_y[:36], all_pred[:36], out_path=os.path.join(config.OUTPUT_DIR, "grid_area_final.png"), class_names=config.AREA_NAMES)
    summarize_classifier(all_y, all_pred, out_dir=config.OUTPUT_DIR, tag="final_rect")
    
    print(f"Final Rectangle Accuracy: {np.mean(np.array(all_y) == np.array(all_pred)):.3f}")

if __name__ == "__main__":
    main()