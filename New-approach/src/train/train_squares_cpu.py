import os, time, random, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.det_dataset import SquaresDetectionDataset, collate_fn
from src.dataio.det_transforms import Compose, ToTensor, RandomHorizontalFlip
from src.models.fasterrcnn_cpu import build_fasterrcnn_cpu
from src.utils.metrics_det import evaluate_ap_by_size
from src.utils.viz import show_prediction

# =========================
# Config (edit paths here)
# =========================
TRAIN_IMG_DIR = "dataset/train_images"  # squares
TRAIN_XML_DIR = "dataset/train_xml"
TEST_IMG_DIR = "dataset/test_images"  # squares
TEST_XML_DIR = "dataset/test_xml"

OUTPUT_DIR = "outputs"
SAVE_CKPT = os.path.join(OUTPUT_DIR, "fasterrcnn_squares_cpu.pt")

EPOCHS = 8
BATCH_SIZE = 1  # CPU-friendly
LR = 0.003
VAL_RATIO = 0.1
SEED = 42
NUM_WORKERS = 0  # keep 0 on Windows
SAVE_VIZ = 10  # number of test images to save with predicted boxes


# =========================
# Utilities
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(pairs, train=False, bs=1):
    if len(pairs) == 0:
        return None
    if train:
        ds = SquaresDetectionDataset(
            pairs, transforms=Compose([ToTensor(), RandomHorizontalFlip(0.5)])
        )
    else:
        ds = SquaresDetectionDataset(pairs, transforms=Compose([ToTensor()]))
    # shuffle only when there are 2+ items
    do_shuffle = train and (len(pairs) > 1)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=do_shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )


# =========================
# Training script
# =========================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    # --- Pair image/XML files
    train_pairs_all = paired_image_xml_list(TRAIN_IMG_DIR, TRAIN_XML_DIR)
    test_pairs = paired_image_xml_list(TEST_IMG_DIR, TEST_XML_DIR)

    if len(train_pairs_all) == 0:
        raise RuntimeError(
            "No training pairs found. Check dataset/train_images & train_xml."
        )
    if len(test_pairs) == 0:
        print(
            "WARNING: No test pairs found. Create dataset/test_images & test_xml if you want test metrics."
        )

    # --- Safe val split from TRAIN (still squares)
    n = len(train_pairs_all)
    if n == 1:
        val_pairs = []
        train_pairs = train_pairs_all
    else:
        n_val = max(1, int(n * VAL_RATIO))
        random.shuffle(train_pairs_all)
        val_pairs = train_pairs_all[:n_val]
        train_pairs = train_pairs_all[n_val:]
        if len(train_pairs) == 0:
            train_pairs = [val_pairs.pop()]  # ensure non-empty train

    print(f"Train={len(train_pairs)} | Val={len(val_pairs)} | Test={len(test_pairs)}")

    # --- DataLoaders
    train_loader = make_loader(train_pairs, train=True, bs=BATCH_SIZE)
    val_loader = make_loader(val_pairs, train=False, bs=BATCH_SIZE)
    test_loader = (
        make_loader(test_pairs, train=False, bs=BATCH_SIZE)
        if len(test_pairs) > 0
        else None
    )

    # --- Model
    model = build_fasterrcnn_cpu(num_classes=2).to(device)  # 1 class + background
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, EPOCHS // 2), gamma=0.1
    )

    # --- Train loop
    best_val_ap = -1.0
    for epoch in range(EPOCHS):
        model.train()
        ep_losses, t0 = [], time.time()
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.item()))
        sched.step()

        # quick val AP (global only) — reuse the by-size evaluator and read the global number
        val_ap = None
        if val_loader is not None:
            metrics, _, _ = evaluate_ap_by_size(
                model,
                val_loader,
                device=device,
                out_dir=OUTPUT_DIR,
                tag=f"val_ep{epoch+1}",
            )
            val_ap = metrics["ap50_global"]
        print(
            f"Epoch {epoch+1:02d} | loss={np.mean(ep_losses):.4f} | val AP@0.5={val_ap if val_ap is not None else 'n/a'} | {time.time()-t0:.1f}s"
        )

        if val_ap is not None and val_ap > best_val_ap:
            best_val_ap = val_ap
            torch.save(model.state_dict(), SAVE_CKPT)
            print(f"  ✓ Saved best checkpoint → {SAVE_CKPT}")

    # --- Load best and do final evaluations (VAL + TEST) with size buckets
    model.load_state_dict(torch.load(SAVE_CKPT, map_location=device))
    if val_loader is not None:
        evaluate_ap_by_size(
            model, val_loader, device=device, out_dir=OUTPUT_DIR, tag="val_final"
        )
    if test_loader is not None:
        evaluate_ap_by_size(
            model, test_loader, device=device, out_dir=OUTPUT_DIR, tag="test_final"
        )

    # --- Save a few prediction overlays from TEST (if present)
    if test_loader is not None and SAVE_VIZ > 0:
        print("Saving example predictions from TEST...")
        model.eval()
        saved = 0
        with torch.no_grad():
            for imgs, targets in test_loader:
                img = imgs[0].to(device)
                pred = model([img])[0]
                out_path = os.path.join(OUTPUT_DIR, f"test_pred_{saved+1}.png")
                show_prediction(
                    img.cpu(), pred, targets[0], score_thr=0.5, save_path=out_path
                )
                print("  saved:", out_path)
                saved += 1
                if saved >= SAVE_VIZ:
                    break

    # --- Write a small run summary JSON
    summary = {
        "epochs": EPOCHS,
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "test_images": len(test_pairs),
        "checkpoint": SAVE_CKPT,
    }
    with open(os.path.join(OUTPUT_DIR, "det_run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. Metrics & plots saved under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
