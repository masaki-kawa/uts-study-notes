# Lecture 3: CNN による画像分類 `Convolutional Neural Networks`

> スライド「Deep Learning - Lecture 3」をベースに、Labノートブックで補足。

---

## 目次

1. [[#1. コンピュータビジョン Computer Vision]]
2. [[#2. デジタル画像の基礎]]
3. [[#3. フィルタと特徴抽出 Filters & Feature Extraction]]
4. [[#4. 畳み込みニューラルネットワーク CNN]]
5. [[#5. CNN の構成要素]]
6. [[#6. PyTorch による CNN 実装]]
7. [[#7. Lab まとめ（Exercise 1〜3）]]

---

## 1. コンピュータビジョン `Computer Vision`

### 定義

> コンピュータビジョンとは、AIの一分野であり、コンピュータがデジタル画像・動画などの視覚入力から意味ある情報を導き出せるようにする技術。

### 主な応用例

| 分野 | 具体例 |
|------|--------|
| 文書認識 | OCR（光学文字認識）|
| 医療 | 医療画像診断 |
| 自動車 | 自動運転・安全システム |
| 小売 | 無人レジ |
| エンターテインメント | モーションキャプチャ、CGI合成 |
| セキュリティ | 監視カメラ、指紋認証・生体認証 |
| 製造 | 品質検査 |

---

## 2. デジタル画像の基礎

### 画像とは何か

コンピュータは人間のように「見る」ことができない。カメラが画像を**ピクセル `pixel`** の行列（マトリクス）に変換することで、デジタル入力として扱える。

### 画像の3つの次元

```
画像 = 幅 (width) × 高さ (height) × チャンネル数 (channels)
```

| 要素 | 説明 |
|------|------|
| **幅・高さ** | ピクセル数で定義（例: 1024×512）。数値が大きいほど高精細 |
| **ピクセル値** | 0〜255 の整数。0 = 黒（低輝度）、255 = 白（高輝度）|
| **チャンネル** | グレースケール = 1ch、カラー（RGB）= 3ch |

### グレースケール vs カラー

```
グレースケール: (H, W, 1)   ← 輝度情報のみ
カラー (RGB) : (H, W, 3)   ← R（赤）、G（緑）、B（青）の3チャンネル
```

> **画像は非構造化データ `unstructured data`** — 年齢・性別のような決まった列（特徴量）がなく、パターンとして情報を含む。

---

## 3. フィルタと特徴抽出 `Filters & Feature Extraction`

### 構造化データ vs 非構造化データ

| 構造化データ | 非構造化データ（画像） |
|------------|-------------------|
| 明確な列・値が存在 | ピクセルの行列 |
| 年齢・性別・購買履歴など | 色・テクスチャ・形状などのパターン |

### フィルタ `Filter` と特徴マップ `Feature Map`

- **フィルタ**: 画像から特定のパターン（エッジ・テクスチャ等）を検出する小さな行列
- **特徴マップ**: フィルタを適用した出力。検出されたパターンが強調された派生画像

```
入力画像 × フィルタ = 特徴マップ
```

### フィルタの例

| フィルタ種類 | 検出するもの |
|------------|------------|
| 垂直フィルタ `Vertical Filter` | 垂直方向のエッジ |
| 水平フィルタ `Horizontal Filter` | 水平方向のエッジ |
| Sobel フィルタ | エッジ全般 |

---

## 4. 畳み込みニューラルネットワーク `CNN`

### 定義

> CNNとは、コンピュータビジョンで広く使われるANNの一種。画像から**自動的・適応的に特徴を学習**するよう設計されており、畳み込み層・プーリング層・全結合層から構成される。

### 通常のNNとの違い

| 通常の Neural Network | CNN |
|----------------------|-----|
| 全入力が全ニューロンに接続 | フィルタがスライドしながら局所的に処理 |
| 画像の空間構造を無視 | 空間的な特徴を保持 |
| パラメータ数が膨大 | 重みの共有で効率的 |

### 畳み込み演算 `Convolution Operation`

3つのステップで構成：

1. **要素積 `element-wise product`**: 入力行列とフィルタを要素ごとに掛け算
2. **総和 `sum`**: 掛け算した要素をすべて足す
3. **ストライド `stride`**: フィルタを指定ピクセル数分スライドして次の位置へ

```
入力行列 A × フィルタ B → 特徴マップ C
```

### ストライド `Stride`

フィルタが入力画像より小さい場合、フィルタをスライドさせながら全ピクセルを処理する。ストライドは**何ピクセルずつ移動するか**を定義する。

### パディング `Padding`

畳み込み後、特徴マップのサイズは入力より小さくなる。これを防ぐために、画像の周囲に **0** を追加する。

```
パディングなし: 出力サイズ < 入力サイズ
パディングあり: 出力サイズ = 入力サイズ（same padding）
```

### プーリング `Pooling`

- 畳み込みは計算コストが高い → 特徴マップのサイズを削減する
- 情報をあまり失わずに次元を圧縮
- 代表的な手法: **Max Pooling**（最大値を取る）、Average Pooling（平均値）

---

## 5. CNN の構成要素

### 全体アーキテクチャ

```
入力画像
  ↓
[畳み込み層 (Conv) + 活性化 (ReLU) + プーリング (MaxPool)] × N回
  ↓  ← 特徴抽出ブロック
フラット化 (Flatten)
  ↓
[全結合層 (FC) + ReLU] × M回
  ↓  ← 分類ブロック
出力層 (Softmax / Sigmoid)
```

### スライド掲載のシンプルなCNN例（参考）

| 層 | 活性化サイズ | パラメータ数 |
|----|------------|------------|
| Input | (32, 32, 3) | 0 |
| CONV1 (f=5, S=1) | (28, 28, 8) | 608 |
| POOL1 | (14, 14, 8) | 0 |
| CONV2 (f=5, S=1) | (10, 10, 16) | 3,216 |
| POOL2 | (5, 5, 16) | 0 |
| FC1 | (120, 1) | 48,120 |
| FC2 | (84, 1) | 10,164 |
| Softmax | (10, 1) | 850 |

---

## 6. PyTorch による CNN 実装

### `nn.Conv2d` のパラメータ

```python
nn.Conv2d(
    in_channels=1,    # 入力チャンネル数（グレースケール=1、RGB=3）
    out_channels=32,  # 出力チャンネル数（フィルタの枚数）
    kernel_size=3,    # フィルタサイズ（3×3）
    stride=1,         # ストライド
    padding=1         # パディング（same padding = 入出力サイズを揃える）
)
```

### CNN クラスの書き方（スライドのテンプレート）

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
```

### データセットの読み込み

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # カラーなら (0.5,0.5,0.5)
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

### DataLoader

```python
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader  = DataLoader(test_data,  batch_size=64, shuffle=True)
```

- `batch_size`: 一度に処理するサンプル数
- `shuffle=True`: エポックごとにデータをランダムに並び替え → 過学習 `overfitting` を抑制

### 学習ループ

```python
for images, labels in train_dataloader:
    outputs = model(images)               # 順伝播 Forward Propagation
    loss = criterion(outputs, labels)     # 損失計算
    optimizer.zero_grad()                 # 勾配リセット（必ず先に）
    loss.backward()                       # 逆伝播 Back Propagation
    optimizer.step()                      # 重み更新
```

> **順番を間違えない**: `forward → loss → zero_grad → backward → step`

### テスト・評価

```python
model.eval()              # 評価モード（Dropout等を無効化）
with torch.no_grad():     # 勾配計算を無効化（メモリ節約・高速化）
    for images, labels in test_dataloader:
        outputs = model(images)
        predicted = torch.max(outputs, 1)[1]
        correct += (predicted == labels).sum().item()

accuracy = correct / total
```

---

## 7. Lab まとめ（Exercise 1〜3）

### 各 Exercise の比較

| | Exercise 1 | Exercise 2 | Exercise 3 |
|---|---|---|---|
| データセット | MNIST | CIFAR-100 | Cats vs Dogs |
| 画像サイズ | 28×28 | 32×32 | 100×100 |
| チャンネル | 1（グレースケール） | 3（RGB）| 3（RGB）|
| クラス数 | 10 | 100 | 2（2値分類）|
| CNN深さ | 2層 | 3層 | 13層（VGG16）|
| 出力活性化 | CrossEntropyLoss | CrossEntropyLoss | Sigmoid |

### 正規化の違い

```python
# グレースケール（チャンネル1つ）
transforms.Normalize((0.5,), (0.5,))

# カラー RGB（チャンネル3つ）
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

### データ拡張 `Data Augmentation`（Exercise 3）

```python
transforms.RandomResizedCrop(100)   # ランダムクロップ
transforms.RandomHorizontalFlip()   # 水平反転
transforms.RandomRotation(40)       # ±40度回転
```

> **訓練データのみ**に適用。テストデータには `Resize` のみ。

### 出力サイズの計算式

$$\text{output} = \left\lfloor \frac{\text{input} - \text{kernel} + 2 \times \text{padding}}{\text{stride}} \right\rfloor + 1$$

```python
def calculate_dim(input_dim, kernel_size, padding, stride):
    return math.floor((input_dim - kernel_size + 2 * padding) / stride) + 1
```

---

## 重要用語まとめ

| 日本語 | 英語 | 説明 |
|--------|------|------|
| 畳み込み層 | `Conv2d` | フィルタで特徴を抽出 |
| プーリング層 | `MaxPool2d` | 次元を削減・過学習抑制 |
| ストライド | `stride` | フィルタの移動量 |
| パディング | `padding` | 境界に0を追加してサイズを保持 |
| 特徴マップ | `feature map` | フィルタ適用後の出力 |
| 全結合層 | `Linear` / `Fully Connected` | 最終分類 |
| 過学習 | `overfitting` | 訓練データに過度に適合 |
| データ拡張 | `data augmentation` | 学習データを人工的に増やす |
| バッチサイズ | `batch size` | 一度に処理するサンプル数 |

---

`#deep-learning` `#CNN` `#PyTorch` `#computer-vision` `#UTS`
