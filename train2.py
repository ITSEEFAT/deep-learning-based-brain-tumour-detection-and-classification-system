import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42
NUM_EPOCHS = 25  # Augmenté car plus d'augmentations
LEARNING_RATE = 0.001

# Classes dures (nécessitent plus d'augmentations)
HARD_CLASSES = [0, 1]  # 0=glioma, 1=meningioma
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Fixer les seeds pour reproductibilité
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================
# AUGMENTATIONS AVANCÉES AVEC ALBUMENTATIONS
# ============================================================

# Augmentations légères (pour classes faciles : notumor, pituitary)
light_transform = A.Compose([
    A.Resize(128, 128),
    A.HorizontalFlip(p=0.5),           # ← changé: RandomHorizontalFlip → HorizontalFlip
    A.RandomRotate90(p=0.3),           # ← celui-ci est correct
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])

# Augmentations lourdes (pour classes difficiles : glioma, meningioma)
heavy_transform = A.Compose([
    A.Resize(128, 128),
    A.HorizontalFlip(p=0.5),           # ← changé
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.4),
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])

# Validation / Test (pas d'augmentation)
eval_transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])

# ============================================================
# DATASET PERSONNALISÉ AVEC AUGMENTATION CIBLÉE
# ============================================================
class AugmentedDataset(torch.utils.data.Dataset):
    """Applique des augmentations différentes selon la classe"""
    
    def __init__(self, dataset, indices, transform_light, transform_heavy, hard_classes, heavy_prob=0.7):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform_light = transform_light
        self.transform_heavy = transform_heavy
        self.hard_classes = set(hard_classes)
        self.heavy_prob = heavy_prob
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        
        # Convertir PIL en numpy pour albumentations
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        # Choisir la transformation selon la classe
        if label in self.hard_classes:
            # Classes difficiles : souvent augmentation lourde
            if random.random() < self.heavy_prob:
                transformed = self.transform_heavy(image=img_np)
            else:
                transformed = self.transform_light(image=img_np)
        else:
            # Classes faciles : uniquement augmentation légère
            transformed = self.transform_light(image=img_np)
        
        return transformed['image'], label

# ============================================================
# MODÈLE (identique à l'original)
# ============================================================
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================
# POIDS DES CLASSES POUR LA LOSS (optionnel mais recommandé)
# ============================================================
def compute_class_weights(dataset, indices):
    """Calcule des poids inverses à la fréquence des classes"""
    labels = []
    for idx in indices:
        _, label = dataset[idx]
        labels.append(label)
    
    unique, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / (counts / len(labels))
    weights = weights / weights.min()  # Normaliser
    
    print(f"  Distribution des classes: {dict(zip([CLASS_NAMES[u] for u in unique], counts))}")
    print(f"  Poids calculés: {dict(zip([CLASS_NAMES[u] for u in unique], [f'{w:.2f}' for w in weights]))}")
    
    return torch.tensor(weights, dtype=torch.float32)

# ============================================================
# FONCTIONS D'ENTRAÎNEMENT
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("ENTRAÎNEMENT AVEC AUGMENTATION CIBLÉE")
    print("=" * 60)
    
    # Chargement du dataset
    source = datasets.ImageFolder('data/Training')
    print(f"\nDataset chargé: {len(source)} images totales")
    print(f"Classes: {source.classes}")
    
    # Split train/val
    val_size = int(VAL_SPLIT * len(source))
    train_size = len(source) - val_size
    
    train_indices, val_indices = random_split(
        range(len(source)), [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Créer les datasets avec augmentation ciblée
    train_dataset = AugmentedDataset(
        source, train_indices, 
        light_transform, heavy_transform, 
        hard_classes=HARD_CLASSES,
        heavy_prob=0.7  # 70% de chance d'avoir augmentation lourde sur classes difficiles
    )
    
    # Validation: pas d'augmentation
    class ValDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = list(indices)
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if isinstance(img, Image.Image):
                img_np = np.array(img)
            else:
                img_np = img
            transformed = self.transform(image=img_np)
            return transformed['image'], label
    
    val_dataset = ValDataset(source, val_indices, eval_transform)
    test_dataset = datasets.ImageFolder('data/Testing')
    
    # Transformer test dataset
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            if isinstance(img, Image.Image):
                img_np = np.array(img)
            else:
                img_np = img
            transformed = self.transform(image=img_np)
            return transformed['image'], label
    
    test_dataset_wrapped = TestDataset(test_dataset, eval_transform)
    
    # Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset_wrapped, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\nTrain: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Poids pour la loss (optionnel)
    class_weights = compute_class_weights(source, train_indices)
    class_weights = class_weights.to(DEVICE)
    
    # Modèle
    model = BrainTumorCNN(num_classes=len(source.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Loss pondérée !
    # Dans la section MAIN, vers ligne 300
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  

    print(f"\nModèle | Device: {DEVICE} | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss pondérée activée | Scheduler: ReduceLROnPlateau")
    
    # Entraînement
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    print(f"\nDémarrage de l'entraînement ({NUM_EPOCHS} epochs)...")
    print("-" * 70)
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model2.pth')
            print(f"  * Nouveau meilleur modele sauvegarde (val_acc={val_acc:.4f})")
        
        scheduler.step(val_loss)  # ReduceLROnPlateau attend la validation loss
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
    
    print("-" * 70)
    print(f"[OK] Entrainement termine | Meilleure Val Accuracy: {best_val_acc:.4f}")
    
    # Test final
    print("\n" + "=" * 60)
    print("ÉVALUATION FINALE SUR TEST SET")
    print("=" * 60)
    
    best_model = BrainTumorCNN(num_classes=len(source.classes)).to(DEVICE)
    best_model.load_state_dict(torch.load('model2.pth'))
    test_loss, test_acc = validate(best_model, test_loader, criterion, DEVICE)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 60)
    
    # Courbes d'entraînement
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=4)
    axes[0].plot(val_losses, label='Val Loss', marker='s', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Progression (avec augmentation ciblée)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, label='Train Accuracy', marker='o', markersize=4)
    axes[1].plot(val_accs, label='Val Accuracy', marker='s', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Progression (avec augmentation ciblée)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Entraînement - Best Val Acc: {best_val_acc:.2%} | Test Acc: {test_acc:.2%}')
    plt.tight_layout()
    plt.savefig('training_curves_augmented.png', dpi=120, bbox_inches='tight')
    plt.show()
    
    print("\n[OK] Courbes sauvegardees dans 'training_curves_augmented.png'")
    print("\n[FILE] Fichier modele sauvegarde: model2.pth")

if __name__ == "__main__":
    main()