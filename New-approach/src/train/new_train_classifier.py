import os, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List, Tuple

from src.setup import new_config_cls as config 
from src.utils.device_utils import select_device
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
from src.dataio.new_cls_dataset_stream import GeometricShapeClassificationDatasetStream 
from src.models.new_resnet import build_resnet_classifier # Now loads MultiTaskResNet
from src.utils.cls_viz import save_confusion_matrix, save_sample_grid
from src.utils.metrics_cls import summarize_classifier


def main():
    # SETUP AND DEVICE INITIALIZATION
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(config.SEED)
    
    # DYNAMIC DEVICE SELECTION (CUDA READY)
    device = select_device(config.DEVICE)
    is_cuda = device.type == "cuda"

    # DATA PREPARATION

    # Training/Validation (Squares Data)
    train_pairs_all = paired_image_xml_list(config.IMG_DIR_TRAIN, config.XML_DIR_ALL)
    val_pairs_all   = paired_image_xml_list(config.IMG_DIR_VAL, config.XML_DIR_ALL)

    # Testing (Rectangles Data)
    test_pairs_all  = paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT)

    # Subsample deterministically using config fractions
    train_pairs = subsample_pairs(train_pairs_all, config.F_TRAIN, seed=config.SEED, max_items=config.MAX_TRAIN_ITEMS)
    val_pairs   = subsample_pairs(val_pairs_all, config.F_VAL, seed=config.SEED)
    test_pairs  = subsample_pairs(test_pairs_all, config.F_TEST, seed=config.SEED)

    print(f"Data Subsets (Used/Total): Train={len(train_pairs)}/{len(train_pairs_all)}, Val={len(val_pairs)}/{len(val_pairs_all)}, Test={len(test_pairs)}")

    # DATASETS AND DATALOADERS (GPU Optimized)
    ds_train = GeometricShapeClassificationDatasetStream(
        train_pairs, canvas=config.CANVAS_SIZE, train=True, use_padding_canvas=config.USE_PADDING_CANVAS
    )
    ds_val = GeometricShapeClassificationDatasetStream(
        val_pairs, canvas=config.CANVAS_SIZE, train=False, use_padding_canvas=config.USE_PADDING_CANVAS
    )
    ds_test = GeometricShapeClassificationDatasetStream(
        test_pairs, canvas=config.CANVAS_SIZE, train=False, use_padding_canvas=config.USE_PADDING_CANVAS
    )
    
    # DataLoaders setup
    train_loader = DataLoader(ds_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=is_cuda)
    val_loader   = DataLoader(ds_val,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=is_cuda)
    test_loader  = DataLoader(ds_test,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=is_cuda)
    
    # MODEL, OPTIMIZER, SCHEDULER
    model = build_resnet_classifier().to(device) 
    
    # Loss functions: CrossEntropy for Area (Classification), MSE for W/H (Regression)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    # TRAINING LOOP (Multi-Task Loss)
    best_area_acc = 0.0
    for ep in range(config.EPOCHS):
        model.train()
        losses = []
        t0 = time.time()
        
        # Load X, Y_AREA (Classification Label), and Y_WH (Regression Target)
        for x, y_area, y_wh in tqdm(train_loader, desc=f"Epoch {ep+1}/{config.EPOCHS}"):
            x = x.to(device, non_blocking=True)
            y_area = y_area.to(device, non_blocking=True)
            y_wh = y_wh.to(device, non_blocking=True) # Regression target - float tensor [W, H]
            
            optimizer.zero_grad()
            area_out, wh_out = model(x) # Model returns Area logits and W/H predictions
            
            # CLASSIFICATION (Area)
            loss_area = criterion_cls(area_out, y_area) 
            
            # REGRESSION (W/H) - Use MSE
            loss_reg = criterion_reg(wh_out, y_wh)
            
            # Sum the losses
            total_loss = loss_area + loss_reg 
            
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())
            
        scheduler.step()

        # VALIDATION
        model.eval()
        area_correct, area_total = 0, 0
        total_mse = 0.0
        
        with torch.no_grad():
            for x, y_area, y_wh in val_loader:
                x = x.to(device, non_blocking=True)
                y_area = y_area.to(device, non_blocking=True)
                y_wh = y_wh.to(device, non_blocking=True)
                
                area_out, wh_out = model(x)
                
                # Area Task Accuracy
                area_pred = area_out.argmax(1)
                area_correct += (area_pred == y_area).sum().item()
                area_total += y_area.size(0)

                # Regression Task MSE
                total_mse += criterion_reg(wh_out, y_wh).item() * y_wh.size(0)
        
        area_acc = area_correct / area_total if area_total else 0.0
        avg_mse = total_mse / area_total if area_total else 0.0
        
        print(f"Epoch {ep+1}: loss={np.mean(losses):.4f} | val_ACC_Area={area_acc:.3f} | val_MSE_WH={avg_mse:.2f} | {time.time()-t0:.1f}s")
        
        if area_acc > best_area_acc: # Save based on the more robust Area task
            best_area_acc = area_acc
            torch.save(model.state_dict(), config.SAVE_CKPT)
            print(f"  ✓ saved best → {config.SAVE_CKPT}")

    print("\nBest validation Area acc:", best_area_acc)

    # FINAL TEST AND REPORTING (Test on Rectangles)
    model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
    model.eval()
    
    all_y_area, all_pred_area = [], []
    all_y_wh_reg, all_pred_wh_reg = [], [] # W/H regression results
    sample_images = []
    total_test_mse = 0.0
    
    with torch.no_grad():
        for x, y_area, y_wh in tqdm(test_loader, desc="Test (Rectangles)"):
            x = x.to(device)
            y_wh = y_wh.to(device)
            area_out, wh_out = model(x)
            
            # Area Task Predictions
            pred_area = area_out.argmax(1)
            all_y_area.extend(y_area.tolist())
            all_pred_area.extend(pred_area.cpu().tolist())

            # W/H Regression Predictions (collecting true and predicted W/H)
            all_y_wh_reg.extend(y_wh.cpu().tolist())
            all_pred_wh_reg.extend(wh_out.cpu().tolist())
            
            # Total Test MSE
            total_test_mse += criterion_reg(wh_out, y_wh).item() * y_wh.size(0)

            if len(sample_images) < 36:
                sample_images.extend(x[:36 - len(sample_images)].cpu())
    
    # Area Test Results
    area_test_acc = np.mean(np.array(all_y_area) == np.array(all_pred_area))
    print(f"\nFinal Test Accuracy (Area Task on Rectangles): {area_test_acc:.3f}")
    summarize_classifier(all_y_area, all_pred_area, out_dir=config.OUTPUT_DIR, tag="test_rectangles_area")
    
    # W/H Regression Results
    avg_test_mse = total_test_mse / len(all_y_area)
    # Calculate Root Mean Squared Error (RMSE) for easier interpretation
    rmse = np.sqrt(avg_test_mse)
    
    print(f"Final Test RMSE (W/H Regression on Rectangles): {rmse:.2f} pixels")
    print(f"Final Test Avg Absolute Error (Conceptual): ~{rmse:.2f} pixels")

    # Visualization (Only Area Classification is plotted)
    cm_path_area = save_confusion_matrix(all_y_area, all_pred_area, 
                                        out_path=os.path.join(config.OUTPUT_DIR, "confmat_cls_area.png"),
                                        class_names=config.AREA_NAMES) # Pass Area names
    print("Saved Area ConfMat:", cm_path_area)
    
    if len(sample_images) > 0:
        grid_path = save_sample_grid(torch.stack(sample_images), 
                                     all_y_area[:len(sample_images)], all_pred_area[:len(sample_images)],
                                     out_path=os.path.join(config.OUTPUT_DIR, "preds_grid_area.png"),
                                     class_names=config.AREA_NAMES) 
        print("Saved Area Grid:", grid_path)
    

if __name__ == "__main__":
    main()