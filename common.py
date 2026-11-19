"""Shared utilities: config, data loading, model building."""
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
DATA_DIR = "data"            # data/<class_name>/*.jpg|png
MODEL_PATH = "models/best_model.pt"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
PATIENCE = 6                 # early stopping
SEED = 42
NUM_WORKERS = 0              # keep 0 on Windows
# ----------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_loaders():
    """Stratified 70/15/15 split of an ImageFolder dataset."""
    full_train = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    full_eval = datasets.ImageFolder(DATA_DIR, transform=eval_tf)
    targets = np.array(full_train.targets)
    idx = np.arange(len(targets))

    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=targets, random_state=SEED)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=targets[temp_idx], random_state=SEED)

    train_ds = Subset(full_train, train_idx)   # with augmentation
    val_ds = Subset(full_eval, val_idx)        # no augmentation
    test_ds = Subset(full_eval, test_idx)

    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    # class weights for imbalance
    counts = Counter(targets[train_idx])
    n_cls = len(full_train.classes)
    weights = torch.tensor(
        [len(train_idx) / (n_cls * counts[i]) for i in range(n_cls)],
        dtype=torch.float)
    return train_loader, val_loader, test_loader, full_train.classes, weights


def build_model(num_classes, pretrained=True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
    return model


def load_trained_model(path=MODEL_PATH):
    ckpt = torch.load(path, map_location=DEVICE)
    classes = ckpt["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, classes