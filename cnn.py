import os
import sys
import time
import copy 
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report


AIRPORT_CLASSES = [
    "pushback",
    "ucak"
]

IMAGE_SIZE: int = 224

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

DEFAULT_BATCH_SIZE: int = 32

DEFAULT_NUM_WORKERS: int = 2

DEFAULT_EPOCHS: int = 20

DEFAULT_LEARNING_RATE: float = 1e-3

CONFIDENCE_THRESHOLD: float = 0.5

TOP_K_PREDICTIONS: int = 5

DEFAULT_OUTPUT_FOLDER: str = "output"

SUPPORTED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

TEXT_COLOR_BGR : Tuple[int, int, int] = (0, 255, 0)

DEFAULT_MODEL_FILENAME: str = "best_model.pth"

def get_device() -> str:
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using Metal Performance Shaders (MPS)")
        return "mps"
    else:
        print("Using CPU")
        return "cpu"

def get_data_transforms(augmentation: bool)-> transforms.Compose:
    normalize_step = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize_step,
            ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize_step,
        ])

def load_datasets(data_dir: str, batch_size: int = DEFAULT_BATCH_SIZE, num_workers: int = DEFAULT_NUM_WORKERS)-> Tuple[DataLoader, DataLoader, List[str]]:
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Train or validation directory not found in {data_dir}")

    print(f"Eğitim verisi okunuyor: {train_dir}")
    train_dataset = ImageFolder(root=train_dir, transform=get_data_transforms(augmentation=True))

    print(f"Doğrulama verisi okunuyor: {val_dir}")
    val_dataset = ImageFolder(root=val_dir, transform=get_data_transforms(augmentation=False))

    class_names = train_dataset.classes

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and validation datasets have different classes.")

    print(f"Bulunan sınıflar: {class_names}")
    print(f"Eğitim örnek sayısı: {len(train_dataset)}")
    print(f"Doğrulama örnek sayısı: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, class_names

def show_batch(loader: DataLoader, class_names: List[str], output_dir: str = DEFAULT_OUTPUT_FOLDER)->None:
    os.makedirs(output_dir, exist_ok=True)
    
    images, labels = next(iter(loader))

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1) # Channel Height Width
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    num_to_show = min(8, images.size(0))
    fig, axes = plt.subplots(2,3,figsize=(12, 8))
    axes = axes.flatten()

    for idx in range(num_to_show):
        img = images[idx] * std + mean  # Unnormalize
        img = img.clamp(0,1).permute(1,2,0).numpy()
        axes[idx].imshow(img)
        axes[idx].set_title(class_names[labels[idx].item()])
        axes[idx].axis('off')

    for idx in range(num_to_show, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "sample_batch.png")
    plt.savefig(save_path)
    print(f"Sample batch saved to {save_path}")
    plt.close(fig)

class CustomCNN(nn.Module):
    def __init__(self,  num_classes: int):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        # 1. Blok: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool
        x = self.relu(self.bn1(self.conv1(x))) # (32,224,224)
        x = self.relu(self.bn2(self.conv2(x))) # (32,224,224)
        x = self.pool(x)                       # (32,112,112)

        # 2. Blok: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool
        x = self.relu(self.bn3(self.conv3(x))) # (64,112,112)
        x = self.relu(self.bn4(self.conv4(x))) # (64,112,112)
        x = self.pool(x)                       # (64,56,56)

        # 3. Blok: Conv -> BN -> ReLU -> Pool
        x = self.relu(self.bn5(self.conv5(x))) # (128,56,56)
        x = self.pool(x)                       # (128,28,28) 

        x = self.global_pool(x)  # Global Average Pooling (128,1,1)

        x = torch.flatten(x, 1)  # Flatten to (128)

        x = self.dropout(x)      # Dropout
        x = self.fc1(x)          # Final fully connected layer to num_classes

        return x

def build_model(model_type: str, num_classes:int, freeze_backbone: bool = False)-> nn.Module:
    model_type = model_type.lower()
    if model_type == "custom":
        print("Building Custom CNN model...")
        return CustomCNN(num_classes=num_classes)

    elif model_type == "resnet18":
        print("Building ResNet18 model...")
        model = models.resnet18(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for parametre in model.parameters():
                parametre.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_type == "efficientnet_b0":
        print("Building EfficientNet-B0 model...")
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for parametre in model.parameters():
                parametre.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif model_type == "mobilenet_v3":
        print("Building MobileNetV3 model...")
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for parametre in model.parameters():
                parametre.requires_grad = False

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    else:
        print(f"Unknown model type '{model_type}'.")
        print("Supported model types: 'custom', 'resnet18', 'efficientnet_b0', 'mobilenet_v3'")
        sys.exit(1)

def count_parameters(model: nn.Module)-> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"%{100 * trainable_params / total_params:.2f} of parameters are trainable.")

def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str,
)-> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        running_correct += (predicted == labels).sum().item()
        running_total += images.size(0)

    epoch_loss = running_loss / running_total
    epoch_accuracy = running_correct / running_total
    return epoch_loss, epoch_accuracy

def validate_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: str,
)-> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            running_correct += (predicted == labels).sum().item()
            running_total += images.size(0)

    epoch_loss = running_loss / running_total
    epoch_accuracy = running_correct / running_total
    return epoch_loss, epoch_accuracy


def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                epochs: int, 
                lr: float, 
                device: str, 
                save_path:str, 
                early_stop_patience: Optional[int]=5)-> Dict[str, List[float]]:

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print(f"Eğitim başlıyor: {epochs} epoch, öğrenme oranı: {lr}, cihaz: {device}")
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                f"Time: {epoch_duration:.2f}s")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"Model kaydedildi: {save_path}") 
            print(f"En iyi doğrulama doğruluğu güncellendi: %{best_val_accuracy*100:.2f}")
            print(f"Epoch {epoch} - Model kaydedildi: {save_path}")
        else:
            epochs_without_improvement += 1

        if early_stop_patience is not None and epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping: {early_stop_patience} epoch boyunca doğrulama doğruluğu iyileşmedi.")
            break
    print("Eğitim tamamlandı.")
    print(f"En iyi doğrulama doğruluğu: %{best_val_accuracy*100:.2f}")
    return history

def plot_training_history(history: Dict[str, List[float]], output_dir: str = DEFAULT_OUTPUT_FOLDER)-> None:
    os.makedirs(output_dir, exist_ok=True)

    epoch_range = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epoch_range, history["train_loss"], label="Train Loss", marker='o')
    ax_loss.plot(epoch_range, history["val_loss"], label="Val Loss", marker='o')
    ax_loss.set_title("Eğitim ve Doğrulama Kayıp Grafiği")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Kayıp (Loss)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epoch_range, history["train_acc"], label="Train Acc", marker='o')
    ax_acc.plot(epoch_range, history["val_acc"], label="Val Acc", marker='o')
    ax_acc.set_title("Eğitim ve Doğrulama Doğrruluk Grafiği")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Doğruluk (Accuracy)")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Eğitim geçmişi grafikleri kaydedildi: {save_path}")

def evaluate_model(
        model: nn.Module,
        loader: DataLoader,
        device: str,
        class_names: List[str],
        output_dir: str = DEFAULT_OUTPUT_FOLDER
)-> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_labels: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0)
    print("\nSınıflandırma Raporu:\n")
    print(report)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Gerçek Sınıf",
        xlabel="Tahmin Edilen Sınıf",
        title="Karışıklık Matrisi (Confusion Matrix)"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max()/2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig)
    print(f"Karışıklık matrisi kaydedildi: {cm_path}")

    return {
        "confusion_matrix": cm,
        "classification_report": report
    }


def load_trained_model(
        model_path: str,
        model_type: str,
        num_classes: int,
        device: str,
)-> nn.Module:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        sys.exit(1)

    model = build_model(model_type=model_type, num_classes=num_classes)

    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model yüklendi: {model_path}")
    return model

def predict_single_image(
        model: nn.Module,
        image_path: str,
        class_names: List[str],
        device: str,
        output_dir: str = DEFAULT_OUTPUT_FOLDER
)-> Optional[List[Tuple[str, float]]]:

    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        print(f"Resim okunamadı: {image_path}")
        return None

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    transform = get_data_transforms(augmentation=False)

    from PIL import Image
    pil_image = Image.fromarray(rgb_image)

    input_tensor = transform(pil_image)

    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=min(TOP_K_PREDICTIONS, len(class_names)))
    top_predictions = [(class_names[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_indices)]

    print(f"{image_path} için tahminler (TOP-{TOP_K_PREDICTIONS} tahmin):\n")
    for rank, (class_name, probability) in enumerate(top_predictions, start=1):
        print(f"{rank}. {class_name}: %{probability*100:.2f}")

    best_class, best_prob = top_predictions[0]
    label_text = f"{best_class}: %{best_prob*100:.1f}"
    cv2.putText(bgr_image, label_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR_BGR, 2, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    output_filename= f"predicted_{Path(image_path).stem}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, bgr_image)
    print(f"Tahmin edilen resim kaydedildi: {output_path}")

    return top_predictions

def predict_folder(
        model: nn.Module,
        folder_path: str,
        class_names: List[str],
        device: str,
        output_dir: str = DEFAULT_OUTPUT_FOLDER
)-> List[Dict[str, object]]:
    if not os.path.isdir(folder_path):
        print(f"Geçersiz klasör yolu: {folder_path}")
        return []

    all_files = sorted(os.listdir(folder_path))
    image_files = [f for f in all_files if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]

    if not image_files:
        print(f"Hiçbir resim dosyası bulunamadı: {folder_path}")
        return []

    print(f"{len(image_files)} resim dosyası bulundu: {folder_path}")

    results: List[Dict[str, object]] = []
    for index, filename in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, filename)
        print(f"\n[{index}/{len(image_files)}] {image_path} için tahmin yapılıyor...")
        top_predictions = predict_single_image(model, image_path, class_names, device, output_dir)

        if top_predictions is not None:
            best_class, best_prob = top_predictions[0]
            results.append({
                'dosya': filename,
                'tahmin': best_class,
                'guven': f"{best_prob*100:.2f}" 
            })
        else:
            results.append({
                'dosya': filename,
                'tahmin': "HATA",
                'guven': "0.00"
            })

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "predictions.csv")

    with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['dosya', 'tahmin', 'guven'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{len(results)} sonuç CSV dosyasına kaydedildi: {csv_path}")
    
    return results

def realtime_webcam_cnn(model: nn.Module, class_names: List[str], device: str)-> None:
    transform = get_data_transforms(augmentation=False)
    from PIL import Image

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam açılamadı.")
        return

    print("Webcam açıldı. Çıkmak için 'q' tuşuna basın.")

    while True:
        ret ,frame = cap.read()
        if not ret:
            print("Webcam görüntüsü alınamadı.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, k=min(TOP_K_PREDICTIONS, len(class_names)))
            top_predictions = [(class_names[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_indices)]
            best_class, best_prob = top_predictions[0]
            main_label = f"{best_class}: %{best_prob*100:.1f}"
            cv2.putText(frame, main_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for rank, (class_name, probability) in enumerate(top_predictions[1:], start=2):
                small_label = f"{rank}. {class_name}: %{probability*100:.1f}"
                y_position = 30 + rank * 30
                cv2.putText(frame, small_label, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR_BGR, 1, cv2.LINE_AA)

            cv2.imshow("Gerçek Zamanlı Tahmin", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Çıkış yapılıyor...")
                break

    cap.release()
    cv2.destroyAllWindows()

def parse_arguments()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN Model Eğitimi ve Tahmin Aracı")
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'eval', 'predict', 'webcam'], help="Çalıştırma modu: 'train', 'eval', 'predict', 'webcam'")
    parser.add_argument("--data-dir", type=str, default="dataset", help="ImageFolder formatındaki veri seti klasörü (/train ve /val içermelidir)")
    parser.add_argument("--model-path", type=str, default=os.path.join(DEFAULT_OUTPUT_FOLDER, DEFAULT_MODEL_FILENAME), help="Kaydedilmiş/Kaydedilecek model dosyası yolu")
    parser.add_argument("--image", type=str, help="(predict modunda) Tek bir resim dosyası için tahmin yapılacaksa resim yolu")
    parser.add_argument("--folder", type=str, help="(predict modunda) Bir klasördeki tüm resimler için tahmin yapılacaksa klasör yolu")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Eğitim için epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Eğitim ve doğrulama için batch boyutu")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Eğitim için öğrenme oranı")
    parser.add_argument("--model-type", type=str, default="custom", choices=['custom', 'resnet18', 'efficientnet_b0', 'mobilenet_v3'], help="Kullanılacak model türü")
    parser.add_argument("--freeze-backbone", action="store_true", help="Transfer Learning için kullanılan Backbone ağırlıklarını dondur (sadece son katmanı eğit)")
    parser.add_argument("--num-classes", type=int, default=None, help="Sınıf sayısı")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_FOLDER, help="Eğitim ve tahmin sonuçlarının kaydedileceği klasör")
    parser.add_argument("--early-stop", type=int, default=0, help="Early stopping için epoch sayısı (varsayılan: 0 ile devre dışı bırakılır)")
    return parser.parse_args()


def main(args: argparse.Namespace)-> None:
    print(50*"=")
    print("CNN TABANLI GÖRÜNTÜ SINIFLANDIRMA ARACI")
    print(50*"=")

    print(f"Çalıştırma modu: {args.mode.upper()}")
    print(50*"=")

    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Kullanılacak cihaz: {device.upper()}")

    if args.mode == "train":
        train_loader, val_loader, class_names = load_datasets(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=DEFAULT_NUM_WORKERS)
        num_classes = len(class_names) if args.num_classes is None else args.num_classes
        model = build_model(model_type=args.model_type, num_classes=num_classes, freeze_backbone=args.freeze_backbone)
        count_parameters(model)

        history = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            epochs=args.epochs, lr=args.lr, device=device,
            save_path=args.model_path, early_stop_patience=5,
        )

        plot_training_history(history, output_dir=args.output_dir)
        best_model = load_trained_model(args.model_path, args.model_type, num_classes, device)
        evaluate_model(best_model, val_loader, device, class_names, output_dir=args.output_dir)

    elif args.mode == "eval":
        _, val_loader, class_names = load_datasets(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=DEFAULT_NUM_WORKERS)
        num_classes = len(class_names)
        model = load_trained_model(args.model_path, args.model_type, num_classes, device)
        evaluate_model(model, val_loader, device, class_names, output_dir=args.output_dir)

    elif args.mode == "predict":
        if args.num_classes is not None:
            class_names = [f"sinif_{i}" for i in range(args.num_classes)]
            print("[UYARI] Sınıf isimleri belirtilmedi, varsayılan olarak sinif_0, sinif_1, ... kullanılıyor.")
        else:
            class_names = AIRPORT_CLASSES
            print(f"[BİLGİ] Varsayılan sınıf isimleri kullanılıyor: {class_names}")

        model = load_trained_model(args.model_path, args.model_type, len(class_names), device)

        if args.image:
            predict_single_image(model, args.image, class_names, device, output_dir=args.output_dir)
        elif args.folder:
            predict_folder(model, args.folder, class_names, device, output_dir=args.output_dir)
        else:
            print("Hata: Tahmin yapmak için --image veya --folder argümanı belirtilmelidir.")
            sys.exit(1)

    elif args.mode == "webcam":
        if args.num_classes is not None:
            class_names = [f"sinif_{i}" for i in range(args.num_classes)]
            print("[UYARI] Sınıf isimleri belirtilmedi, varsayılan olarak sinif_0, sinif_1, ... kullanılıyor.")
        else:
            class_names = AIRPORT_CLASSES
            print(f"[BİLGİ] Varsayılan sınıf isimleri kullanılıyor: {class_names}")

        model = load_trained_model(args.model_path, args.model_type, len(class_names), device)
        realtime_webcam_cnn(model, class_names, device)
    print(50*"=")
    print("İşlem tamamlandı.")
    print(50*"=")

if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
