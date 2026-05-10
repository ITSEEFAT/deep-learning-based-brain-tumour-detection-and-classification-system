import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ── Config ────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE  = 32
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
COLORS      = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
NUM_EPOCHS  = 25  # matches train2

# ── Model Architecture (matches train2.py exactly) ────────────
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
            nn.Linear(512, 256),         nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ── Load model2.pth ───────────────────────────────────────────
print("Loading model2.pth ...")
model = BrainTumorCNN(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load('model2.pth', map_location=DEVICE, weights_only=True))
model.eval()
print(f"Model loaded | Device: {DEVICE} | Params: {sum(p.numel() for p in model.parameters()):,}")

# ── Test DataLoader ───────────────────────────────────────────
eval_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder('data/Testing', transform=eval_tf)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Test set: {len(test_dataset)} images | Classes: {test_dataset.classes}\n")

# ── Collect predictions ───────────────────────────────────────
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

y_true  = np.array(all_labels)
y_pred  = np.array(all_preds)
y_probs = np.array(all_probs)
y_true_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))

# ══════════════════════════════════════════════════════════════
# 1. MÉTRIQUES GLOBALES
# ══════════════════════════════════════════════════════════════
acc       = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall    = recall_score(y_true, y_pred, average='weighted')
f1        = f1_score(y_true, y_pred, average='weighted')

print(f"\n{'='*60}")
print(f"  ÉVALUATION COMPLÈTE — MODEL 2 — TEST SET")
print(f"{'='*60}")
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"{'='*60}")

print("\nRapport de classification par classe :")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# Per-class metrics
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class    = recall_score(y_true, y_pred, average=None)
f1_per_class        = f1_score(y_true, y_pred, average=None)
auc_per_class       = [
    auc(*roc_curve(y_true_bin[:, i], y_probs[:, i])[:2])
    for i in range(len(CLASS_NAMES))
]

print(f"\n{'='*60}")
print(f"  RÉSUMÉ PAR CLASSE")
print(f"{'='*60}")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:>12} | P={precision_per_class[i]:.3f} | R={recall_per_class[i]:.3f} | F1={f1_per_class[i]:.3f} | AUC={auc_per_class[i]:.3f}")
print(f"{'─'*60}")
print(f"  {'Weighted':>12} | P={precision:.3f} | R={recall:.3f} | F1={f1:.3f}")
print(f"  Accuracy globale : {acc*100:.2f}%")
print(f"{'='*60}")

errors_idx = np.where(y_pred != y_true)[0]
print(f"\n── Analyse des erreurs ──────────────────────────────")
print(f"  Erreurs totales : {len(errors_idx)} / {len(y_true)}")
print(f"  Taux d'erreur   : {len(errors_idx)/len(y_true)*100:.2f}%\n")
error_pairs = {}
for i in errors_idx:
    pair = (CLASS_NAMES[y_true[i]], CLASS_NAMES[y_pred[i]])
    error_pairs[pair] = error_pairs.get(pair, 0) + 1
print("  Confusions les plus fréquentes :")
for (true_cls, pred_cls), count in sorted(error_pairs.items(), key=lambda x: -x[1]):
    print(f"    {true_cls:>12} -> {pred_cls:<12} : {count} fois")

# ══════════════════════════════════════════════════════════════
# FIGURE 2_1 — Matrice de confusion (brute + normalisée)
# ══════════════════════════════════════════════════════════════
cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title('Matrice de Confusion (brute) — Model 2', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Prediction'); axes[0].set_ylabel('Verite')

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
axes[1].set_title('Matrice de Confusion (normalisee) — Model 2', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Prediction'); axes[1].set_ylabel('Verite')

plt.tight_layout()
plt.savefig('fig2_1_confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n✓ fig2_1_confusion_matrix.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 2_2 — Métriques par classe (bar chart)
# ══════════════════════════════════════════════════════════════
x     = np.arange(len(CLASS_NAMES))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - width, precision_per_class, width, label='Precision', color='steelblue',   alpha=0.85)
b2 = ax.bar(x,         recall_per_class,    width, label='Recall',    color='darkorange',  alpha=0.85)
b3 = ax.bar(x + width, f1_per_class,        width, label='F1-Score',  color='forestgreen', alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
ax.set_title('Precision / Recall / F1-Score par classe — Model 2', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_2_metrics_per_class.png', dpi=120, bbox_inches='tight')
plt.show()
print("✓ fig2_2_metrics_per_class.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 2_3 — Courbes ROC (une par classe)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    youden_idx = np.argmax(tpr - fpr)

    axes[i].plot(fpr, tpr, color=color, lw=2.5, label=f'AUC = {roc_auc:.4f}')
    axes[i].fill_between(fpr, tpr, alpha=0.12, color=color)
    axes[i].plot([0, 1], [0, 1], 'k--', lw=1.2, label='Aleatoire')
    axes[i].plot(fpr[youden_idx], tpr[youden_idx], 'o', color=color, markersize=9,
                 zorder=5, label=f'Seuil optimal = {thresholds[youden_idx]:.2f}')
    axes[i].set_title(f'ROC - {cls.upper()} — Model 2', fontsize=13, fontweight='bold', color=color)
    axes[i].set_xlabel('FPR', fontsize=11); axes[i].set_ylabel('TPR', fontsize=11)
    axes[i].legend(fontsize=10, loc='lower right')
    axes[i].grid(alpha=0.3); axes[i].set_xlim([0,1]); axes[i].set_ylim([0,1.02])

plt.suptitle('Courbes ROC par Classe (One-vs-Rest) — Model 2', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_3_roc_per_class.png', dpi=130, bbox_inches='tight')
plt.show()
print("✓ fig2_3_roc_per_class.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 2_4 — Courbes Precision-Recall (une par classe)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    p_curve, r_curve, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
    ap       = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    baseline = y_true_bin[:, i].mean()

    axes[i].plot(r_curve, p_curve, color=color, lw=2.5, label=f'AP = {ap:.4f}')
    axes[i].fill_between(r_curve, p_curve, alpha=0.12, color=color)
    axes[i].axhline(y=baseline, color='gray', linestyle='--', lw=1.2,
                    label=f'Baseline = {baseline:.2f}')
    axes[i].set_title(f'Precision-Recall - {cls.upper()} — Model 2', fontsize=13,
                      fontweight='bold', color=color)
    axes[i].set_xlabel('Recall', fontsize=11); axes[i].set_ylabel('Precision', fontsize=11)
    axes[i].legend(fontsize=10, loc='upper right')
    axes[i].grid(alpha=0.3); axes[i].set_xlim([0,1]); axes[i].set_ylim([0,1.05])

plt.suptitle('Courbes Precision-Recall par Classe — Model 2', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_4_pr_per_class.png', dpi=130, bbox_inches='tight')
plt.show()
print("✓ fig2_4_pr_per_class.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 2_5 — Distribution des probabilités par classe
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    probs_cls = y_probs[:, i]
    tp_probs  = probs_cls[y_true == i]
    fp_probs  = probs_cls[y_true != i]

    axes[i].hist(fp_probs, bins=30, alpha=0.65, color='#95A5A6',
                 label='Autres classes', density=True)
    axes[i].hist(tp_probs, bins=30, alpha=0.75, color=color,
                 label=f'Classe {cls}', density=True)
    axes[i].axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Seuil = 0.5')
    axes[i].set_title(f'Distribution scores - {cls.upper()} — Model 2', fontsize=13,
                      fontweight='bold', color=color)
    axes[i].set_xlabel('Probabilite predite', fontsize=11)
    axes[i].set_ylabel('Densite', fontsize=11)
    axes[i].legend(fontsize=10); axes[i].grid(alpha=0.3); axes[i].set_xlim([0,1])

plt.suptitle('Distribution des Probabilites par Classe — Model 2', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_5_prob_distributions.png', dpi=130, bbox_inches='tight')
plt.show()
print("✓ fig2_5_prob_distributions.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 2_6 — Dashboard récapitulatif global
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 13))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

# A: Toutes ROC
ax_roc = fig.add_subplot(gs[0, 0])
for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{cls} ({auc_per_class[i]:.3f})')
ax_roc.plot([0,1],[0,1],'k--',lw=1)
ax_roc.set_title('Courbes ROC', fontweight='bold')
ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
ax_roc.legend(fontsize=9, loc='lower right'); ax_roc.grid(alpha=0.3)

# B: Toutes PR
ax_pr = fig.add_subplot(gs[0, 1])
for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    p, r, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    ax_pr.plot(r, p, color=color, lw=2, label=f'{cls} (AP={ap:.3f})')
ax_pr.set_title('Courbes Precision-Recall', fontweight='bold')
ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
ax_pr.legend(fontsize=9, loc='upper right'); ax_pr.grid(alpha=0.3)

# C: Radar chart
ax_radar = fig.add_subplot(gs[0, 2], polar=True)
metrics_radar = np.array([precision_per_class, recall_per_class, f1_per_class, auc_per_class])
metric_labels = ['Precision', 'Recall', 'F1', 'AUC']
angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
angles += angles[:1]
for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    values = metrics_radar[:, i].tolist() + [metrics_radar[0, i]]
    ax_radar.plot(angles, values, 'o-', color=color, lw=2, label=cls)
    ax_radar.fill(angles, values, alpha=0.07, color=color)
ax_radar.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=11)
ax_radar.set_ylim(0, 1)
ax_radar.set_title('Radar des metriques', fontweight='bold', pad=18)
ax_radar.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.35, 1.1))

# D: Confusion matrix normalisée
ax_cm = fig.add_subplot(gs[1, 0])
cm_norm2 = confusion_matrix(y_true, y_pred, normalize='true')
sns.heatmap(cm_norm2, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax_cm, cbar=False)
ax_cm.set_title('Matrice de Confusion (normalisee)', fontweight='bold')
ax_cm.set_xlabel('Prediction'); ax_cm.set_ylabel('Verite')

# E: F1 vs AUC bar chart
ax_bar = fig.add_subplot(gs[1, 1])
x_pos = np.arange(len(CLASS_NAMES))
w = 0.35
ax_bar.bar(x_pos - w/2, f1_per_class,  w, label='F1-Score', color=COLORS, alpha=0.8)
ax_bar.bar(x_pos + w/2, auc_per_class, w, label='AUC',      color=COLORS, alpha=0.45,
           edgecolor=COLORS, linewidth=1.5)
ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels(CLASS_NAMES, rotation=15)
ax_bar.set_ylim(0, 1.1); ax_bar.set_ylabel('Score')
ax_bar.set_title('F1-Score vs AUC par classe', fontweight='bold')
ax_bar.legend(); ax_bar.grid(axis='y', alpha=0.3)

# F: Résumé texte
ax_txt = fig.add_subplot(gs[1, 2])
ax_txt.axis('off')
summary = f"  RESUME GLOBAL — MODEL 2\n{'─'*30}\n\n"
summary += f"  Accuracy  : {acc:.4f}\n"
summary += f"  Precision : {precision:.4f}\n"
summary += f"  Recall    : {recall:.4f}\n"
summary += f"  F1-Score  : {f1:.4f}\n\n"
summary += f"{'─'*30}\n\n"
for i, cls in enumerate(CLASS_NAMES):
    summary += f"  {cls:<12} F1={f1_per_class[i]:.3f} AUC={auc_per_class[i]:.3f}\n"
ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#F0F4F8',
                      edgecolor='#BDC3C7', linewidth=1.5))

plt.suptitle('Dashboard - Evaluation Complete BrainTumorCNN — Model 2',
             fontsize=17, fontweight='bold', y=1.01)
plt.savefig('fig2_6_dashboard.png', dpi=140, bbox_inches='tight')
plt.show()
print("✓ fig2_6_dashboard.png")

# ── Récapitulatif ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  FICHIERS GENERES — MODEL 2")
print(f"{'='*55}")
for fname, desc in [
    ("fig2_1_confusion_matrix.png",    "Matrices de confusion brute + normalisee"),
    ("fig2_2_metrics_per_class.png",   "Precision / Recall / F1 par classe"),
    ("fig2_3_roc_per_class.png",       "Courbes ROC (1 graphe / classe)"),
    ("fig2_4_pr_per_class.png",        "Courbes Precision-Recall / classe"),
    ("fig2_5_prob_distributions.png",  "Distributions des probabilites"),
    ("fig2_6_dashboard.png",           "Dashboard recapitulatif global"),
]:
    print(f"  + {fname:<38} {desc}")
print(f"{'='*55}")