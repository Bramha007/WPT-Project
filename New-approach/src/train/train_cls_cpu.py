import random, numpy as np, torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.cls_dataset import SquaresClassificationDataset
from src.utils.metrics_cls import summarize_classifier
from src.models.resnet_cls import build_resnet_classifier
from src.utils.cls_viz import (
    save_confusion_matrix,
    save_sample_grid,
    print_classification_report,
)


# --- config ---
TRAIN_IMG_DIR = "dataset/train_images"
TRAIN_XML_DIR = "dataset/train_xml"
TEST_IMG_DIR = "dataset/test_images"
TEST_XML_DIR = "dataset/test_xml"

SAVE_CKPT = "outputs/resnet_cls.pt"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
VAL_RATIO = 0.1
SEED = 42
# -------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(SEED)
    device = torch.device("cpu")

    train_pairs = paired_image_xml_list(TRAIN_IMG_DIR, TRAIN_XML_DIR)
    test_pairs = paired_image_xml_list(TEST_IMG_DIR, TEST_XML_DIR)

    full_dataset = SquaresClassificationDataset(train_pairs, canvas=224, train=True)
    n_val = max(1, int(len(full_dataset) * VAL_RATIO))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = build_resnet_classifier(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # val
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f} val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_CKPT)
            print(f"  ✓ saved best → {SAVE_CKPT}")

    print("Best val acc:", best_acc)

    # evaluate on test set (squares)
    if len(test_pairs) > 0:
        test_ds = SquaresClassificationDataset(test_pairs, canvas=224, train=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
        model.load_state_dict(torch.load(SAVE_CKPT, map_location=device))
        model.eval()
        correct, total = 0, 0

        all_y, all_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                all_y.extend(y.tolist())
                all_pred.extend(pred.tolist())
        acc = correct / total if total > 0 else 0
        print("Test acc:", acc)
        print("Confusion matrix:")
        print(confusion_matrix(all_y, all_pred))

    # inside main(), after training finishes:
    print("Best val acc:", best_acc)

    # evaluate on test set (squares)
    if len(test_pairs) > 0:
        test_ds = SquaresClassificationDataset(test_pairs, canvas=224, train=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model.load_state_dict(torch.load(SAVE_CKPT, map_location=device))
        model.eval()

        correct, total = 0, 0
        all_y, all_pred = [], []
        sample_images = []  # collect a batch of images for the grid

        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                all_y.extend(y.tolist())
                all_pred.extend(pred.tolist())

                # stash up to 36 images for the grid
                if len(sample_images) < 36:
                    # ensure we don't exceed 36
                    needed = 36 - len(sample_images)
                    sample_images.extend(x[:needed])

        _ = summarize_classifier(all_y, all_pred, out_dir="outputs", tag="test")
        print("Saved cls metrics JSON under outputs/")

        test_acc = correct / total if total > 0 else 0
        print(f"\nTest acc: {test_acc:.3f}")

        # Visuals + detailed report
        cm_path = save_confusion_matrix(
            all_y, all_pred, out_path="outputs/confmat_cls.png"
        )
        print("Saved confusion matrix →", cm_path)

        if len(sample_images) > 0:
            grid_path = save_sample_grid(
                torch.stack(sample_images),
                all_y[: len(sample_images)],
                all_pred[: len(sample_images)],
                out_path="outputs/preds_grid.png",
                max_samples=36,
            )
            print("Saved prediction grid →", grid_path)

        print_classification_report(all_y, all_pred)

    else:
        print(
            "No test set found — add dataset/test_images & test_xml to generate visualizations."
        )


if __name__ == "__main__":
    main()
