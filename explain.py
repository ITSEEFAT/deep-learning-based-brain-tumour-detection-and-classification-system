import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ============================================================
# MODÈLE (identique à train2.py)
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
# GRAD-CAM IMPLEMENTATION
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_activation(module, input, output):
            self.activations = output.detach()
        
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)
    
    def generate(self, image_tensor, class_idx=None):
        self.model.eval()
        
        # Forward pass (must be outside no_grad for backward to work)
        output = self.model(image_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Compute CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx

# ============================================================
# CHARGEMENT MODÈLE
# ============================================================
print("=" * 60)
print("G RAD - C A M   A N A L Y S I S")
print("=" * 60)

print("Loading model2.pth ...")
model = BrainTumorCNN(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load('model2.pth', map_location=DEVICE, weights_only=True))
model.eval()
print(f"Model loaded | Device: {DEVICE} | Params: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# TEST DATALOADER
# ============================================================
eval_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder('data/Testing', transform=eval_tf)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Test set: {len(test_dataset)} images | Classes: {test_dataset.classes}\n")

# ============================================================
# FONCTION GRAD-CAM VISUALISATION
# ============================================================
def visualize_gradcam(model, test_loader, class_names, device, num_images=8):
    """Visualise Grad-CAM sur des images bien et mal classées"""
    
    target_layer = model.features[-3]
    gradcam = GradCAM(model, target_layer)

    images_to_show = []
    labels_to_show = []
    preds_to_show = []
    cams_to_show = []

    # Step 1: collect candidates under no_grad
    collected = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(images)):
                if len(collected) >= num_images:
                    break
                is_correct = (preds[i] == labels[i]).item()
                if len(collected) < num_images // 2 and is_correct:
                    collected.append((images[i].cpu(), labels[i].item(), preds[i].item(), True))
                elif len(collected) >= num_images // 2 and not is_correct:
                    collected.append((images[i].cpu(), labels[i].item(), preds[i].item(), False))

            if len(collected) >= num_images:
                break

    if len(collected) == 0:
        print("  Aucune image trouvée pour l'analyse Grad-CAM")
        return

    # Step 2: generate CAMs outside no_grad so gradients flow
    for img_cpu, true_lbl, pred_lbl, is_correct in collected:
        img_tensor = img_cpu.unsqueeze(0).to(device)
        target_cls = true_lbl if is_correct else pred_lbl
        cam, _ = gradcam.generate(img_tensor, target_cls)
        images_to_show.append(img_cpu)
        labels_to_show.append(true_lbl)
        preds_to_show.append(pred_lbl)
        cams_to_show.append(cam)

    # Visualisation
    rows = 2
    cols = num_images // rows
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()

    for idx in range(len(images_to_show)):
        img = images_to_show[idx].numpy().transpose(1, 2, 0)
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)

        cam_resized = cv2.resize(cams_to_show[idx], (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        axes[idx].imshow(img)
        axes[idx].imshow(heatmap, alpha=0.6)
        axes[idx].axis('off')

        is_correct = (labels_to_show[idx] == preds_to_show[idx])
        color = 'green' if is_correct else 'red'
        title = f"True: {class_names[labels_to_show[idx]]}\nPred: {class_names[preds_to_show[idx]]}"
        axes[idx].set_title(title, color=color, fontsize=10, fontweight='bold')

    for idx in range(len(images_to_show), len(axes)):
        axes[idx].axis('off')

    plt.suptitle("Grad-CAM: Zones activées par le modèle\n(🟢 Bien classé | 🔴 Mal classé)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig7_gradcam_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ fig7_gradcam_analysis.png sauvegardé")

# ============================================================
# ANALYSE SPÉCIFIQUE CONFUSIONS GLIOMA/MENINGIOMA
# ============================================================
def analyze_hard_classes_gradcam(model, test_loader, class_names, device):
    """Analyse spécifique des confusions glioma vs meningioma"""

    target_layer = model.features[-3]
    gradcam = GradCAM(model, target_layer)
    hard_cases = []

    # Step 1: collect candidates under no_grad
    collected = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(images)):
                true_cls = labels[i].item()
                pred_cls = preds[i].item()
                if (true_cls == 0 and pred_cls == 1) or (true_cls == 1 and pred_cls == 0):
                    collected.append((images[i].cpu(), true_cls, pred_cls))
                if len(collected) >= 6:
                    break
            if len(collected) >= 6:
                break

    # Step 2: generate CAMs outside no_grad
    for img_cpu, true_cls, pred_cls in collected:
        img_tensor = img_cpu.unsqueeze(0).to(device)
        cam, _ = gradcam.generate(img_tensor, pred_cls)
        hard_cases.append({
            'image': img_cpu,
            'true_label': true_cls,
            'pred_label': pred_cls,
            'cam': cam
        })

    if len(hard_cases) > 0:
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()

        for idx, case in enumerate(hard_cases):
            img = case['image'].numpy().transpose(1, 2, 0)
            img = img * 0.5 + 0.5
            img = np.clip(img, 0, 1)

            cam_resized = cv2.resize(case['cam'], (img.shape[1], img.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

            axes[idx].imshow(img)
            axes[idx].imshow(heatmap, alpha=0.6)
            axes[idx].axis('off')

            title = f"True: {class_names[case['true_label']]} → Pred: {class_names[case['pred_label']]}"
            axes[idx].set_title(title, color='red', fontsize=10, fontweight='bold')

        for idx in range(len(hard_cases), len(axes)):
            axes[idx].axis('off')

        plt.suptitle("Analyse Grad-CAM: Cas de confusion Glioma ↔ Méningiome\n(rouge = région activée menant à l'erreur)",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('fig8_hard_cases_glioma_meningioma.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ fig8_hard_cases_glioma_meningioma.png sauvegardé")
    else:
        print("  Aucune confusion glioma↔meningioma trouvée")

# ============================================================
# EXÉCUTION PRINCIPALE
# ============================================================
print("\n🔍 Génération des visualisations Grad-CAM...\n")

visualize_gradcam(model, test_loader, CLASS_NAMES, DEVICE, num_images=8)

analyze_hard_classes_gradcam(model, test_loader, CLASS_NAMES, DEVICE)

print("\n" + "=" * 60)
print("✅ Analyse Grad-CAM terminée avec succès !")
print("📁 Fichiers générés:")
print("   + fig7_gradcam_analysis.png")
print("   + fig8_hard_cases_glioma_meningioma.png")
print("=" * 60)