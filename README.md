# CNN Tabanlı Görüntü Sınıflandırma Aracı

PyTorch ile yazılmış, tek dosyalık (`cnn.py`) bir görüntü sınıflandırma aracıdır. Aynı betik ile:

- Sıfırdan yazılmış bir CNN'i veya ImageNet ağırlıklı hazır modelleri (ResNet18, EfficientNet-B0, MobileNetV3) **eğitebilir**,
- Eğitilmiş modeli doğrulama setinde **değerlendirebilir** (sınıflandırma raporu + karışıklık matrisi),
- Tek bir resim veya bir klasördeki tüm resimler için **tahmin** yapabilir (sonuçlar CSV'ye yazılır),
- Webcam görüntüsü üzerinde **gerçek zamanlı** tahmin yapabilir.

Varsayılan sınıflar havalimanı senaryosuna göredir: `pushback` ve `ucak`.

---

## İçindekiler

1. [Özellikler](#özellikler)
2. [Proje Yapısı](#proje-yapısı)
3. [Kurulum](#kurulum)
4. [Veri Seti Yapısı](#veri-seti-yapısı)
5. [Kullanım](#kullanım)
6. [Komut Satırı Argümanları](#komut-satırı-argümanları)
7. [Kodun Çalışma Mantığı](#kodun-çalışma-mantığı)
8. [Model Mimarileri](#model-mimarileri)
9. [Eğitim Süreci](#eğitim-süreci)
10. [Çıktı Dosyaları](#çıktı-dosyaları)
11. [Sonuçları Yorumlama](#sonuçları-yorumlama)
12. [Bilinen Sınırlamalar](#bilinen-sınırlamalar)
13. [Sık Karşılaşılan Sorunlar](#sık-karşılaşılan-sorunlar)

---

## Özellikler

| Özellik | Açıklama |
|---|---|
| Dört çalışma modu | `train`, `eval`, `predict`, `webcam` |
| Dört model seçeneği | `custom`, `resnet18`, `efficientnet_b0`, `mobilenet_v3` |
| Transfer learning | `--freeze-backbone` ile yalnızca son katman eğitilebilir |
| Otomatik cihaz seçimi | CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU sırasıyla |
| Veri artırma | Eğitimde rastgele kırpma, çevirme, döndürme, renk oynatma |
| Öğrenme oranı planlayıcı | Adam + Cosine Annealing |
| En iyi modeli kaydetme | Doğrulama doğruluğu en yüksek epoch'un ağırlıkları saklanır |
| Early stopping | Doğrulama doğruluğu belirli sayıda epoch iyileşmezse eğitim durur |
| Raporlama | Eğitim grafikleri, sınıflandırma raporu, karışıklık matrisi |
| Toplu tahmin | Klasördeki tüm resimler için tahmin + `predictions.csv` |
| Gerçek zamanlı tahmin | Webcam üzerinde ilk 5 tahmin ekranda gösterilir |

---

## Proje Yapısı

```
cnn2/
├── cnn.py              # Tüm kod (eğitim, değerlendirme, tahmin, webcam)
├── requirements.txt    # Gerekli Python paketleri
├── README.md           # Bu dosya
├── dataset/            # (Repoya dahil değil) Eğitim ve doğrulama verisi
│   ├── train/
│   └── val/
└── output/             # (Repoya dahil değil) Çalışma sırasında otomatik oluşur
    ├── best_model.pth
    ├── training_history.png
    ├── confusion_matrix.png
    ├── predictions.csv
    └── predicted_*.jpg
```

---

## Kurulum

### Gereksinimler

- Python 3.9 veya üzeri
- (İsteğe bağlı) NVIDIA GPU + CUDA ya da Apple Silicon (MPS desteği)

### Adımlar

```bash
# 1. Sanal ortam oluştur
python3 -m venv venv

# 2. Sanal ortamı etkinleştir (macOS / Linux)
source venv/bin/activate
# Windows için: venv\Scripts\activate

# 3. Bağımlılıkları kur
pip install -r requirements.txt
```

### Kullanılan Kütüphaneler

| Paket | Kullanım amacı |
|---|---|
| `torch`, `torchvision` | Model, veri yükleme, dönüşümler, hazır modeller |
| `numpy` | Karışıklık matrisi çiziminde eksen düzenlemesi |
| `opencv-python` | Resim okuma/yazma, webcam, resmin üzerine metin yazma |
| `Pillow` | OpenCV görüntüsünü torchvision dönüşümlerine uygun formata çevirme |
| `matplotlib` | Eğitim grafikleri ve karışıklık matrisi (arka planda `Agg` ile, pencere açmadan) |
| `scikit-learn` | `confusion_matrix` ve `classification_report` |

---

## Veri Seti Yapısı

Araç, torchvision'ın `ImageFolder` formatını bekler. Her sınıf için bir klasör oluşturulmalı ve `train` ile `val` altında **aynı sınıf isimleri** bulunmalıdır:

```
dataset/
├── train/
│   ├── pushback/
│   │   ├── img001.jpg
│   │   └── ...
│   └── ucak/
│       ├── img101.jpg
│       └── ...
└── val/
    ├── pushback/
    └── ucak/
```

Önemli noktalar:

- Sınıf etiketleri klasör isimlerinden otomatik çıkarılır ve **alfabetik sıraya** göre numaralandırılır (`pushback` → 0, `ucak` → 1).
- `train` ve `val` içindeki sınıf listesi birbirinden farklıysa kod `ValueError` ile durur.
- `train` veya `val` klasörü yoksa `FileNotFoundError` verilir.
- **Veri sızıntısına dikkat:** Aynı videodan/sahneden alınmış birbirine çok benzeyen kareler hem `train`'e hem `val`'e düşerse doğrulama başarısı olduğundan yüksek görünür. Bölmeyi video/sahne bazında yapmanız önerilir.

---

## Kullanım

### 1. Eğitim (`train`)

```bash
# Sıfırdan yazılmış CNN
python cnn.py --mode train --data-dir dataset --epochs 20 --model-type custom

# Daha düşük öğrenme oranı ile
python cnn.py --mode train --data-dir dataset --epochs 20 --model-type custom --lr 3e-4

# Transfer learning (tüm ağ ince ayar)
python cnn.py --mode train --data-dir dataset --model-type resnet18 --lr 1e-4

# Transfer learning (sadece son katman)
python cnn.py --mode train --data-dir dataset --model-type resnet18 --freeze-backbone
```

Eğitim bittiğinde sırasıyla şunlar otomatik yapılır:

1. Eğitim geçmişi grafiği kaydedilir.
2. En iyi model diskten yeniden yüklenir.
3. Doğrulama setinde değerlendirme yapılıp rapor ve karışıklık matrisi üretilir.

### 2. Değerlendirme (`eval`)

Daha önce eğitilmiş modeli doğrulama setinde tekrar değerlendirir:

```bash
python cnn.py --mode eval --data-dir dataset --model-path output/best_model.pth --model-type custom
```

### 3. Tahmin (`predict`)

```bash
# Tek resim
python cnn.py --mode predict --image test/ornek.jpg --model-type custom

# Klasördeki tüm resimler
python cnn.py --mode predict --folder test/ --model-type custom
```

- Tek resim için en olası sınıflar terminale yazılır ve etiketli resim `output/predicted_<dosya_adı>.jpg` olarak kaydedilir.
- Klasör için ek olarak `output/predictions.csv` oluşturulur.

### 4. Webcam (`webcam`)

```bash
python cnn.py --mode webcam --model-type custom
```

Çıkmak için pencere seçiliyken **`q`** tuşuna basın.

---

## Komut Satırı Argümanları

| Argüman | Tür | Varsayılan | Açıklama |
|---|---|---|---|
| `--mode` | str | **zorunlu** | `train`, `eval`, `predict`, `webcam` |
| `--data-dir` | str | `dataset` | `train/` ve `val/` içeren veri seti klasörü |
| `--model-path` | str | `output/best_model.pth` | Modelin kaydedileceği / yükleneceği dosya |
| `--image` | str | – | `predict` modunda tek resim yolu |
| `--folder` | str | – | `predict` modunda klasör yolu |
| `--epochs` | int | `20` | Eğitim epoch sayısı |
| `--batch-size` | int | `32` | Batch boyutu |
| `--lr` | float | `1e-3` | Başlangıç öğrenme oranı |
| `--model-type` | str | `custom` | `custom`, `resnet18`, `efficientnet_b0`, `mobilenet_v3` |
| `--freeze-backbone` | flag | kapalı | Hazır modellerde gövdeyi dondurur, yalnızca son katmanı eğitir |
| `--num-classes` | int | `None` | Sınıf sayısı (aşağıdaki nota bakın) |
| `--output-dir` | str | `output` | Grafik, rapor ve tahmin çıktılarının klasörü |
| `--early-stop` | int | `0` | Early stopping sabır değeri (bkz. [Bilinen Sınırlamalar](#bilinen-sınırlamalar)) |

> **`--num-classes` hakkında:** `train` modunda verilmezse sınıf sayısı veri setinden otomatik bulunur. `predict` ve `webcam` modlarında verilirse sınıf isimleri `sinif_0`, `sinif_1`, ... olarak üretilir; verilmezse koddaki `AIRPORT_CLASSES` listesi (`pushback`, `ucak`) kullanılır.

---

## Kodun Çalışma Mantığı

### Sabitler

Dosyanın başında tüm ayarlar sabit olarak tanımlıdır:

| Sabit | Değer | Anlamı |
|---|---|---|
| `AIRPORT_CLASSES` | `["pushback", "ucak"]` | Tahmin modlarında kullanılan sınıf isimleri |
| `IMAGE_SIZE` | `224` | Modele verilen resim boyutu (224×224) |
| `IMAGENET_MEAN / STD` | ImageNet değerleri | Normalizasyon (hazır modellerle uyum için) |
| `DEFAULT_BATCH_SIZE` | `32` | Varsayılan batch boyutu |
| `DEFAULT_NUM_WORKERS` | `2` | DataLoader işçi sayısı |
| `DEFAULT_EPOCHS` | `20` | Varsayılan epoch sayısı |
| `DEFAULT_LEARNING_RATE` | `1e-3` | Varsayılan öğrenme oranı |
| `TOP_K_PREDICTIONS` | `5` | Gösterilecek en yüksek olasılıklı tahmin sayısı |
| `SUPPORTED_IMAGE_EXTENSIONS` | `.jpg .jpeg .png .bmp .tiff .webp` | Klasör tahmininde taranan uzantılar |

### Fonksiyonlar

#### `get_device()`
Kullanılabilir en hızlı cihazı seçer: önce CUDA, sonra Apple MPS, yoksa CPU.

#### `get_data_transforms(augmentation)`
- **`augmentation=True` (eğitim):** `RandomResizedCrop(224, scale=(0.8, 1.0))` → `RandomHorizontalFlip(0.5)` → `RandomRotation(10°)` → `ColorJitter` → `ToTensor` → `Normalize`
- **`augmentation=False` (doğrulama/tahmin):** `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize`

Eğitimdeki rastgele dönüşümler modeli ezberden korumaya yardım eder; doğrulama ve tahminde sonuçların tekrarlanabilir olması için sabit dönüşüm kullanılır.

#### `load_datasets(data_dir, batch_size, num_workers)`
`train/` ve `val/` klasörlerini `ImageFolder` ile okur, sınıf listelerinin aynı olduğunu doğrular, iki `DataLoader` döndürür. Eğitim loader'ı karıştırır (`shuffle=True`), doğrulama loader'ı karıştırmaz.

#### `show_batch(loader, class_names, output_dir)`
Bir batch'ten örnek resimleri (normalizasyonu geri alarak) `output/sample_batch.png` olarak kaydeder. Veri artırmanın nasıl göründüğünü kontrol etmek için faydalıdır. Not: `main` içinde otomatik çağrılmaz, gerektiğinde elle çağrılabilir.

#### `CustomCNN`
Sıfırdan yazılmış küçük bir CNN (aşağıda ayrıntılı).

#### `build_model(model_type, num_classes, freeze_backbone)`
İstenen modeli oluşturur. Hazır modellerde ImageNet ağırlıkları (`IMAGENET1K_V1`) yüklenir ve son sınıflandırma katmanı `num_classes` çıkışlı yeni bir `Linear` katmanla değiştirilir. `freeze_backbone=True` ise önce tüm parametreler dondurulur, sonra yeni eklenen son katman eğitilebilir kalır.

#### `count_parameters(model)`
Toplam ve eğitilebilir parametre sayısını ve eğitilebilir oranını yazdırır.

#### `train_one_epoch` / `validate_one_epoch`
Bir epoch boyunca eğitim / doğrulama yapar ve ortalama **loss** ile **accuracy** döndürür. Doğrulamada `model.eval()` ve `torch.no_grad()` kullanılır (Dropout kapanır, BatchNorm sabit istatistik kullanır, gradyan hesaplanmaz).

#### `train_model(...)`
Ana eğitim döngüsü (ayrıntıları [Eğitim Süreci](#eğitim-süreci) bölümünde).

#### `plot_training_history(history, output_dir)`
Sol tarafta loss, sağ tarafta accuracy eğrilerini (train ve val) çizip `training_history.png` olarak kaydeder.

#### `evaluate_model(model, loader, device, class_names, output_dir)`
Tüm doğrulama setini tahmin eder, `classification_report` (precision / recall / f1-score / support) yazdırır ve karışıklık matrisini `confusion_matrix.png` olarak kaydeder.

#### `load_trained_model(model_path, model_type, num_classes, device)`
Model dosyasının varlığını kontrol eder, aynı mimariyi yeniden kurar, kayıtlı ağırlıkları yükler ve `eval()` moduna alır.

#### `predict_single_image(...)`
1. Resmi OpenCV ile okur (BGR), RGB'ye çevirir, PIL görüntüsüne dönüştürür.
2. Doğrulama dönüşümlerini uygular, batch boyutu ekler.
3. `softmax` ile olasılıkları hesaplar, ilk `TOP_K` sınıfı seçer.
4. Terminale yazdırır, en yüksek tahmini resmin sol üstüne yazıp `predicted_<ad>.jpg` olarak kaydeder.

#### `predict_folder(...)`
Klasördeki desteklenen uzantılı tüm resimleri alfabetik sırayla işler. Okunamayan resimler `HATA` olarak işaretlenir. Sonuçlar `predictions.csv` dosyasına yazılır.

#### `realtime_webcam_cnn(...)`
Webcam'i açar, her karede aynı ön işleme ve tahmin adımlarını uygular. En iyi tahmin büyük, diğer tahminler küçük yazıyla ekranda gösterilir. `q` tuşu ile kapanır.

#### `parse_arguments()` ve `main(args)`
Komut satırı argümanlarını okur; seçilen moda göre yukarıdaki fonksiyonları sırayla çağırır.

### Akış Şeması

```
python cnn.py --mode ...
        │
        ▼
   get_device()
        │
 ┌──────┼───────────────┬─────────────────┐
 ▼      ▼               ▼                 ▼
train   eval          predict           webcam
 │      │               │                 │
 │      │               │                 └─► realtime_webcam_cnn
 │      │               ├─► predict_single_image
 │      │               └─► predict_folder ─► predictions.csv
 │      └─► load_trained_model ─► evaluate_model
 │
 ├─► load_datasets ─► build_model ─► train_model
 ├─► plot_training_history
 └─► load_trained_model ─► evaluate_model
```

---

## Model Mimarileri

### `custom` – CustomCNN

Üç bloktan oluşan, yaklaşık **140 bin parametreli** (2 sınıf için 140.322) hafif bir ağdır.

| Blok | Katmanlar | Çıkış boyutu |
|---|---|---|
| Giriş | – | 3 × 224 × 224 |
| Blok 1 | Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool | 32 × 112 × 112 |
| Blok 2 | Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool | 64 × 56 × 56 |
| Blok 3 | Conv(64→128) → BN → ReLU → MaxPool | 128 × 28 × 28 |
| Global Avg Pool | AdaptiveAvgPool2d(1×1) | 128 × 1 × 1 |
| Sınıflandırıcı | Flatten → Dropout(0.5) → Linear(128→num_classes) | num_classes |

Tasarım tercihleri:

- **3×3 evrişimler + padding=1:** Boyutu koruyup yalnızca `MaxPool` ile küçültür.
- **BatchNorm:** Eğitimi hızlandırır ve kararlılaştırır.
- **Global Average Pooling:** Büyük bir tam bağlantılı katman yerine kullanıldığı için parametre sayısı çok düşüktür.
- **Dropout (0.5):** Aşırı öğrenmeyi azaltır.

### `resnet18`
ImageNet ile önceden eğitilmiş ResNet-18. `fc` katmanı `num_classes` çıkışlı yeni bir katmanla değiştirilir.

### `efficientnet_b0`
ImageNet ağırlıklı EfficientNet-B0. `classifier[1]` katmanı değiştirilir.

### `mobilenet_v3`
ImageNet ağırlıklı MobileNetV3-Large. `classifier[3]` katmanı değiştirilir. Mobil/gömülü cihazlar için hafif bir seçenektir.

> Hazır modeller ilk çalıştırmada ağırlıkları internetten indirir.
> Tahmin/değerlendirme sırasında `--model-type`, eğitimde kullanılanla **aynı** olmalıdır; aksi halde ağırlıklar yüklenemez.

---

## Eğitim Süreci

| Bileşen | Seçim |
|---|---|
| Kayıp fonksiyonu | `CrossEntropyLoss` |
| Optimizer | `Adam` (yalnızca `requires_grad=True` olan parametreler) |
| LR planlayıcı | `CosineAnnealingLR(T_max=epochs)` – öğrenme oranı kosinüs eğrisiyle 0'a doğru azalır |
| Model seçimi | Doğrulama doğruluğu (`val_acc`) en yüksek olan epoch'un ağırlıkları kaydedilir |
| Early stopping | `val_acc` belirli sayıda epoch üst üste iyileşmezse eğitim durdurulur |

Her epoch'ta sırasıyla:

1. Eğitim setinde bir tur (`train_one_epoch`),
2. Doğrulama setinde bir tur (`validate_one_epoch`),
3. `scheduler.step()`,
4. Geçmişe (history) kayıt ve ekrana özet satırı,
5. `val_acc` yeni rekor ise model kaydı.

Örnek terminal çıktısı:

```
Epoch 4/20 - Train Loss: 0.1170, Train Acc: 0.9655 - Val Loss: 0.0386, Val Acc: 1.0000 - Time: 37.92s
Model kaydedildi: output/best_model.pth
En iyi doğrulama doğruluğu güncellendi: %100.00
```

---

## Çıktı Dosyaları

| Dosya | Üreten mod | İçerik |
|---|---|---|
| `output/best_model.pth` | `train` | En iyi epoch'un model ağırlıkları (`state_dict`) |
| `output/training_history.png` | `train` | Train/val loss ve accuracy grafikleri |
| `output/confusion_matrix.png` | `train`, `eval` | Karışıklık matrisi |
| `output/predicted_<ad>.jpg` | `predict` | Üzerine en iyi tahmin yazılmış resim |
| `output/predictions.csv` | `predict --folder` | `dosya`, `tahmin`, `guven` sütunları |

`predictions.csv` örneği (Excel'de Türkçe karakterler doğru açılsın diye `utf-8-sig` ile kaydedilir):

```csv
dosya,tahmin,guven
ucak_01.jpg,ucak,98.42
pushback_07.jpg,pushback,91.15
bozuk.png,HATA,0.00
```

---

## Sonuçları Yorumlama

**Eğitim grafiği (`training_history.png`)**

- Train ve val loss birlikte düşüyor, birbirine yakınsa → sağlıklı öğrenme.
- Train loss düşerken val loss yükseliyorsa → **overfitting** (aşırı öğrenme).
- Val eğrisi epoch'tan epoch'a çok zıplıyorsa → öğrenme oranı yüksek olabilir, val seti küçük olabilir; `--lr 3e-4` veya `1e-4` deneyin.
- Val başarısı train'in üstündeyse → eğitimde veri artırma olduğu için normal olabilir; yine de val setinin "kolay" olup olmadığını (veri sızıntısı) kontrol edin.

**Sınıflandırma raporu**

- **Precision:** Modelin "bu sınıf" dediklerinin ne kadarı gerçekten o sınıf.
- **Recall:** Gerçekte o sınıfta olanların ne kadarı bulundu.
- **F1-score:** Precision ve recall'un dengeli ortalaması.
- **Support:** O sınıfa ait doğrulama örneği sayısı.

**%100 doğruluk ne anlama gelir?** Küçük bir doğrulama setinde (ör. 206 resim) %100 çoğu zaman gerçek genelleme başarısı değildir. En iyi epoch'u seçtiğiniz veri ile sonucu raporladığınız veri aynı olduğundan sonuç iyimserdir. Gerçek başarıyı görmek için eğitimde ve epoch seçiminde hiç kullanılmamış, farklı sahne/gün/kamera açısından çekilmiş görüntülerle `predict` modunu çalıştırın.

---

## Bilinen Sınırlamalar

- **`--early-stop` argümanı bağlı değil:** `main` içinde `train_model`'e `early_stop_patience=5` sabit olarak verilir; komut satırındaki değer kullanılmaz. Düzeltmek için:
  ```python
  early_stop_patience=args.early_stop if args.early_stop > 0 else None,
  ```
- **Ayrı test seti yok:** Kod yalnızca `train` ve `val` kullanır. Güvenilir bir son ölçüm için bağımsız bir `test` seti ayırmanız önerilir.
- **Sınıf isimleri `predict`/`webcam` modlarında koda gömülü:** Farklı bir veri setiyle çalışırken `AIRPORT_CLASSES` listesini güncelleyin ve sırayı **alfabetik** tutun (ImageFolder ile aynı).
- **`--num-classes` tutarlılığı:** `train` modunda `--num-classes` verilirse `eval` modu bunu kullanmaz (sınıf sayısını veri setinden alır). Tutarsız değer girmeyin.
- **Kullanılmayan öğeler:** `copy`, `torchvision` importları ve `CONFIDENCE_THRESHOLD` sabiti şu an kullanılmıyor, silinebilir.
- **`load_trained_model` içinde ulaşılamayan satır:** `raise` sonrasındaki `sys.exit(1)` hiçbir zaman çalışmaz, silinebilir.
- **Webcam yalnızca 0 numaralı kamerayı açar** (`cv2.VideoCapture(0)`).
- **Webcam modunda kare başına tahmin yapılır;** CPU'da düşük FPS görülebilir.

---

## Sık Karşılaşılan Sorunlar

| Hata / Belirti | Olası neden ve çözüm |
|---|---|
| `error: the following arguments are required: --mode` | `--mode` verilmemiş. Örn: `python cnn.py --mode train` |
| `Train or validation directory not found` | `--data-dir` yolu yanlış ya da `train/` `val/` klasörleri yok |
| `Train and validation datasets have different classes` | `train` ve `val` içindeki sınıf klasör isimleri uyuşmuyor |
| `Model dosyası bulunamadı` | `--model-path` yanlış veya henüz eğitim yapılmamış |
| `size mismatch` / `Missing key(s) in state_dict` | `--model-type` eğitimdekiyle farklı |
| `Webcam açılamadı` | Kamera izni verilmemiş (macOS: Sistem Ayarları → Gizlilik → Kamera) veya başka uygulama kullanıyor |
| Eğitim çok yavaş | `--batch-size` değerini ayarlayın, GPU/MPS kullanıldığından emin olun (çıktının başındaki `Using ...` satırına bakın) |
| Val sonuçları çok oynak | `--lr 3e-4` ya da `1e-4`, veya `--model-type resnet18` deneyin |

---

## Önerilen Çalışma Akışı

1. Veri setini `train/` ve `val/` olarak, mümkünse sahne/video bazında ayırın.
2. `custom` modelle hızlı bir eğitim yapıp grafiği inceleyin.
3. Val eğrisi oynaksa öğrenme oranını düşürün veya `resnet18` deneyin.
4. Eğitimde hiç görülmemiş yeni görüntülerle `predict --folder` çalıştırıp gerçek başarıyı ölçün.
5. Memnun kaldığınızda `webcam` ile gerçek zamanlı testi yapın.
