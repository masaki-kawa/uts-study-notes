# Lab 3: CNNによる画像分類 `Image Classification with PyTorch`

## 概要

| 項目 | 内容 |
|------|------|
| テーマ | 畳み込みニューラルネットワーク `Convolutional Neural Network (CNN)` |
| フレームワーク | PyTorch |
| Exercise 1 | MNIST（手書き数字・多クラス分類） |
| Exercise 2 | CIFAR-100（100クラス物体・多クラス分類） |
| Exercise 3 | Cats vs Dogs（2クラス分類）・VGG16 |

---

## 1. CNNの基本構造

### 全体的な流れ

```
入力画像
  ↓
畳み込み層 (Conv2d) + 活性化関数 (ReLU)
  ↓
プーリング層 (MaxPool2d)  ← ここまでを複数回繰り返す
  ↓
フラット化 (Flatten)
  ↓
全結合層 (Linear) + 活性化関数 (ReLU)
  ↓
出力層 (Linear)
```

### 主要コンポーネント

| コンポーネント | クラス | 役割 |
|--------------|--------|------|
| 畳み込み層 | `nn.Conv2d` | 特徴マップの抽出 |
| 活性化関数 | `nn.ReLU` | 非線形変換・勾配消失を防ぐ |
| プーリング層 | `nn.MaxPool2d` | 空間次元の削減・過学習抑制 |
| 全結合層 | `nn.Linear` | 最終的な分類 |
| 損失関数（多クラス） | `nn.CrossEntropyLoss` | 多クラス分類の誤差計算 |
| 損失関数（2クラス） | `nn.BCELoss` / `nn.Sigmoid` | 2値分類の誤差計算 |
| 最適化 | `torch.optim.Adam` | 重みの更新 |

---

## 2. 出力サイズの計算

畳み込みやプーリング後のサイズは以下の式で計算できる：

$$\text{output\_dim} = \left\lfloor \frac{\text{input\_dim} - \text{kernel\_size} + 2 \times \text{padding}}{\text{stride}} \right\rfloor + 1$$

```python
import math

def calculate_dim(input_dim, kernel_size, padding, stride):
    return math.floor((input_dim - kernel_size + 2 * padding) / stride) + 1
```

### 例: MNIST (28×28) の場合

| 層 | 入力サイズ | 出力サイズ |
|----|-----------|-----------|
| Conv2d(kernel=3, padding=1) | 28×28 | 28×28 |
| MaxPool2d(2, 2) | 28×28 | 14×14 |
| Conv2d(kernel=3, padding=1) | 14×14 | 14×14 |
| MaxPool2d(2, 2) | 14×14 | 7×7 |
| → fc1 の入力次元 | 64 ch × 7 × 7 | = **3136** |

---

## 3. Exercise 1: MNIST（手書き数字）

### データセット

- **画像サイズ**: 28×28 グレースケール（チャンネル数 = 1）
- **クラス数**: 10（0〜9）
- **分割比**: train 70% / val 30% / test（別途）

### データ準備

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # グレースケールは平均・標準偏差が1つ
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
```

> **ポイント**: `Normalize((mean,), (std,))` でピクセル値を `[-1, 1]` にスケーリングし、学習を安定させる。

### CNNアーキテクチャ (CNN_MNIST)

```python
class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

### アーキテクチャまとめ

| 層 | 設定 | 出力 |
|----|------|------|
| Conv2d | in=1, out=32, kernel=3, padding=1 | 32×28×28 |
| MaxPool2d | 2×2 | 32×14×14 |
| Conv2d | in=32, out=64, kernel=3, padding=1 | 64×14×14 |
| MaxPool2d | 2×2 | 64×7×7 |
| Linear (fc1) | 64×7×7=3136 → 128 | 128 |
| Linear (fc2) | 128 → 10 | **10クラス** |

### 学習ループ

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for images, labels in train_dataloader:
        predicted_outputs = model(images)          # 順伝播
        loss = criterion(predicted_outputs, labels) # 損失計算
        optimizer.zero_grad()                       # 勾配リセット
        loss.backward()                             # 逆伝播
        optimizer.step()                            # 重み更新
```

> **学習ループの順番**: `forward → loss → zero_grad → backward → step`

### 検証・テスト

```python
with torch.no_grad():  # 検証・テスト時は勾配計算を無効化
    for images, labels in val_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 最大スコアのクラスを取得
```

---

## 4. Exercise 2: CIFAR-100（100クラス物体）

### データセット

- **画像サイズ**: 32×32 カラー（チャンネル数 = 3: RGB）
- **クラス数**: 100
- **特徴**: 画像サイズが小さいため、5エポックでは精度が低い

### 正規化の違い（グレースケール vs カラー）

```python
# MNIST（グレースケール）
transforms.Normalize((0.5,), (0.5,))

# CIFAR-100（カラー RGB）
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # R, G, B それぞれに指定
```

### CNNアーキテクチャ (CNN_CIFAR) — 3層構成

| 層 | 設定 | 出力サイズ |
|----|------|----------|
| Conv2d | in=3, out=32, kernel=3, padding=1 | 32×32×32 |
| MaxPool2d | 2×2 | 32×16×16 |
| Conv2d | in=32, out=64, kernel=3, padding=1 | 64×16×16 |
| MaxPool2d | 2×2 | 64×8×8 |
| Conv2d | in=64, out=128, kernel=3, padding=1 | 128×8×8 |
| MaxPool2d | 2×2 | 128×4×4 |
| Linear (fc1) | 128×4×4=2048 → 512 | 512 |
| Linear (fc2) | 512 → 100 | **100クラス** |

---

## 5. Exercise 3: Cats vs Dogs（VGG16）

### データセット

- **画像サイズ**: 100×100 カラー（リサイズ後）
- **クラス数**: 2（cat / dog）
- **分割比**: train 70% / val 15% / test 15%

### データ拡張 `Data Augmentation`

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(100),   # ランダムクロップ
    transforms.RandomHorizontalFlip(),   # 水平反転
    transforms.RandomRotation(40),       # ±40度回転
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((100, 100)),       # テスト時はリサイズのみ
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

> **ポイント**: データ拡張は**訓練データのみ**に適用する。テストデータには適用しない。

### VGG16アーキテクチャ

VGG16は13層の畳み込み層 + 3層の全結合層で構成される深いCNN。

| ブロック | 構成 | プーリング後サイズ |
|---------|------|----------------|
| Block 1 | Conv(3→64) × 2 + MaxPool | 64×50×50 |
| Block 2 | Conv(64→128) × 2 + MaxPool | 128×25×25 |
| Block 3 | Conv(128→256) × 3 + MaxPool | 256×12×12 |
| Block 4 | Conv(256→512) × 3 + MaxPool | 512×6×6 |
| Block 5 | Conv(512→512) × 3 + MaxPool | 512×3×3 |
| fc1 | 512×3×3=4608 → 4096 | — |
| fc2 | 4096 → 4096 | — |
| fc3 | 4096 → 1 + Sigmoid | **2クラス** |

### 2値分類の出力

```python
nn.Linear(4096, 1),
nn.Sigmoid()  # 出力を [0, 1] に変換 → 0.5以上で dog、未満で cat
```

> 多クラス分類は `CrossEntropyLoss`、2値分類は `Sigmoid + BCELoss` の組み合わせが基本。

### ImageFolder でのデータ読み込み

```python
train_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(dataset_dir, 'train'),
    transform=train_transform
)
```

> `ImageFolder` はフォルダ名をクラスラベルとして自動認識する。
> ディレクトリ構造: `train/cats/`, `train/dogs/`

---

## 6. 学習のベストプラクティス

| 項目 | 内容 |
|------|------|
| 再現性 | `torch.manual_seed(42)` でシードを固定 |
| バッチサイズ | 64（一般的な設定） |
| 最適化 | `Adam(lr=0.001)` が安定しやすい |
| 評価時 | `model.eval()` + `torch.no_grad()` を必ず使う |
| データ拡張 | 訓練データのみに適用 |
| 学習曲線 | loss と accuracy を epoch ごとに記録・可視化する |

---

## 7. よく使うコードパターン

### DataLoader の作成

```python
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_dataloader  = DataLoader(test_data,     batch_size=batch_size, shuffle=False)
```

### 精度の計算

```python
_, predicted = torch.max(outputs, 1)  # 多クラス
correct += (predicted == labels).sum().item()
accuracy = correct / total
```

### 混同行列 `Confusion Matrix`

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, predicted_labels)
```

---

## 関連タグ

`#deep-learning` `#CNN` `#PyTorch` `#image-classification` `#UTS`
