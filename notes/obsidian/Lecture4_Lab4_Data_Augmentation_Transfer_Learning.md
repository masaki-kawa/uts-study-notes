# Lecture 4: データ拡張・転移学習・事前学習済みモデル `Data Augmentation, Transfer Learning & Pre-Trained Models`

> **対応**: Lecture 4 ↔ Lab 4 | スライド: `Deep Learning - Lecture 4.pptx.pdf` | Colab: `Deep_Learning_Lab4_Exercise1_Solutions.ipynb`, `Deep_Learning_Lab4_Exercise2_Solutions.ipynb`, `Deep_Learning_Lab4_Exercise3_Solutions.ipynb`

---

## 目次

1. [[#1. データ拡張 Data Augmentation]]
2. [[#2. ImageNet]]
3. [[#3. 主要CNNアーキテクチャ]]
4. [[#4. 転移学習とファインチューニング Transfer Learning & Fine-Tuning]]
5. [[#5. PyTorchの事前学習済みモデル Pre-Trained Models in PyTorch]]
6. [[#6. PyTorchコールバック Callbacks]]
7. [[#重要用語まとめ]]
8. [[#復習ポイント]]
9. [[#混同しやすいポイント]]
10. [[#理解チェック問題]]

---

## 1. データ拡張 `Data Augmentation`

### 過学習 `Overfitting` の問題

> **過学習（`overfitting`）**: モデルが訓練データに特化したパターンを学習し、未知のデータ（検証・テストセット、本番データ）に対して汎化できない状態。

$$Loss = Error(y, \hat{y}) + \lambda \sum_{i=1}^{N} |w_i| \quad \text{(L1正則化)}$$
$$Loss = Error(y, \hat{y}) + \lambda \sum_{i=1}^{N} w_i^2 \quad \text{(L2正則化)}$$

**コンピュータビジョンでの具体例**:
- 訓練データ: 小型犬の顔画像のみ
- 未知データ: 大型犬・全身・屋外の画像
- → モデルが訓練データの分布しか学習できていない

**主な原因**: 訓練・検証・テスト各セットのデータ分布が異なること。

### データ拡張の概念

> 過学習を防ぐには、実世界をより良く代表する**多様な入力画像**をモデルに与える必要がある。
> 新しい画像を収集するか、**既存画像から新しい画像を生成**（データ拡張）する。

1枚の元画像から複数のバリエーション（反転・回転・クロップなど）を生成し、訓練データの多様性を人工的に増やす。

### `torchvision.transforms v2` API

**> Lecture → Lab**: スライドの変換一覧 = Colabの `transforms.Compose([...])` に対応。

| カテゴリ | 主な変換関数 | 説明 |
|---------|------------|------|
| Resizing | `v2.Resize(size)` | 指定サイズにリサイズ |
| Cropping | `v2.RandomCrop(size)` | ランダム位置でクロップ |
| Cropping | `v2.RandomResizedCrop(size)` | ランダムクロップ＋リサイズ |
| Cropping | `v2.CenterCrop(size)` | 中央でクロップ |
| Flip | `v2.RandomHorizontalFlip(p)` | 水平反転（確率p） |
| Flip | `v2.RandomVerticalFlip(p)` | 垂直反転（確率p） |
| Color | `v2.Grayscale()` | グレースケール変換 |
| Color | `v2.ColorJitter(brightness, hue)` | 輝度・色相などをランダム変化 |
| Others | `v2.RandomRotation(degrees)` | ランダム回転 |
| Others | `v2.RandomAffine(degrees, ...)` | アフィン変換 |
| Others | `v2.GaussianBlur(kernel_size)` | ガウシアンブラー |

**基本的な使用例**:

```python
# [概念: データ拡張パイプライン] 複数の変換をチェーンで適用
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),  # ランダムクロップ+リサイズ
    v2.RandomHorizontalFlip(p=0.5),                         # 50%の確率で水平反転
    v2.ToDtype(torch.float32, scale=True),                  # float32に変換
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),                # ImageNet統計で正規化
])
img = transforms(img)
```

### 各変換の詳細

**水平反転（`Horizontal Flipping`）**
- 画像のピクセルを水平方向に反転
- `v2.Compose([v2.RandomHorizontalFlip(p=1)])`

**垂直反転（`Vertical Flipping`）**
- 画像のピクセルを垂直方向に反転
- `v2.Compose([v2.RandomVerticalFlip(p=1)])`

**ランダム回転（`Random Rotation`）**
- 画像をランダムな角度で回転
- `v2.Compose([v2.RandomRotation(degrees=(0, 180))])`

**ランダムクロップ（`Random Crop`）**
- 画像をランダムな位置で切り取る
- `v2.Compose([v2.RandomCrop(size=(128, 128))])`

**カラージッター（`Color Jitter`）**
- 輝度・コントラスト・彩度・色相をランダムに変化させる
- `v2.Compose([v2.ColorJitter(brightness=.5, hue=.3)])`

---

## 2. ImageNet

### ImageNetデータセット

> **ImageNet**: 1400万枚以上の画像を含む大規模データセット。画像分類・物体検出用にアノテーション済み。2007年からFei-Fei Liらが整備。

### ILSVRC競技会

**ILSVRC（`ImageNet Large Scale Visual Recognition Challenge`）**:
- 毎年開催される画像分類・物体検出の国際競技会
- 研究者がカスタムアーキテクチャを訓練し性能を評価
- アーキテクチャの詳細と事前学習済み重みを公開

**ILSVRC Top-1精度の推移**（主要モデル）:

| 年 | モデル | Top-1精度 |
|----|-------|----------|
| 2011 | SIFT + FVs | ~50% |
| 2012 | AlexNet | ~63% |
| 2016 | Inception V3 | ~74% |
| 2017 | ResNeXt-101 | ~80% |
| 2020 | FixResNeXt-101 | ~88% |

### ImageNetの2つのタスク

| タスク | 説明 | 出力形式 |
|--------|------|---------|
| 画像分類（`Image Classification`） | Class IDとClass Name対応 | クラスラベル |
| 物体検出（`Object Detection`） | バウンディングボックス座標 | `ImageId, PredictionString` |

---

## 3. 主要CNNアーキテクチャ

### AlexNet

> Alex Krizhevsky ら（トロント大学）が2012年に発表。ILSVRC 2012優勝。

**アーキテクチャ**:
- 畳み込み層（`convolutional layers`）: 5層
- 全結合層（`fully-connected layers`）: 3層
- マックスプーリング（`max pooling`）
- ドロップアウト（`dropout`）

**特徴・主なイノベーション**:
- データ拡張（`data augmentation`）を初めて導入
- `ReLU`活性化関数の採用
- マックスプーリングの活用
- トレーニング環境: 2×Nvidia GTX 580 3GB GPU、5〜6日間

| 層 | フィルタ数 | カーネルサイズ | ストライド | パディング |
|----|-----------|-------------|---------|---------|
| Conv1 | 96 | 11×11 | 4 | 0 |
| Conv2 | 256 | 5×5 | 1 | 2 |
| Conv3 | 384 | 3×3 | 1 | 1 |
| Conv4 | 384 | 3×3 | 1 | 1 |
| Conv5 | 256 | 3×3 | 1 | 1 |
| FC1 | 4096ニューロン | - | - | - |
| FC2 | 4096ニューロン | - | - | - |
| Output | 1000ニューロン | - | - | - |

- 学習可能パラメータ: 約6000万（`~60 Million`）
- Top-5テストエラー率: 15.3%

### VGG

> Karen Simonyan・Andrew Zisserman（オックスフォード大VGGグループ）が2015年に発表。

**アーキテクチャ**:
- 畳み込み層: 13層
- 全結合層: 3層

**構造の特徴**:
- 5ブロック構成
- 各ブロック: 3×3フィルタの畳み込み層2〜3層 + 2×2マックスプーリング（ストライド2）+ ReLU
- **各ブロック後**: 特徴マップのサイズ÷2、フィルタ数×2

| バージョン | 層数 | パラメータ数 |
|-----------|------|-----------|
| VGG-16 | 16層 | 約1億3800万 |
| VGG-19 | 19層 | 約1億4400万 |

**評価**: 理解・改変が容易だが、パラメータ数が多く計算コストが高い。

### Inception（GoogLeNet）

> Christian Szegedy ら（Google）が2014年に発表。ILSVRC 2014優勝（Top-5エラー率6.7%）。

**目標**: 複雑なパターンを理解するためにより深いモデルが必要。ただし:
- モデルが大きくなるほど過学習しやすい（特に訓練データが少ない場合）
- パラメータ増加 → 計算リソースも増加

**Inceptionモジュール（ナイーブ版）**:
- 同じ入力に対して1×1、3×3、5×5の畳み込みと3×3マックスプーリングを**並列**に適用
- 出力をチャンネル次元で結合（`channel concatenation`）

**Inceptionモジュール（次元削減版）**:

```
前の層の出力 (28×28×192)
├── 1×1 Conv → 64ch
├── 1×1 Conv (96ch) → 3×3 Conv → 128ch   # 1×1 でチャンネル削減してから3×3
├── 1×1 Conv (16ch) → 5×5 Conv → 32ch   # 同様に5×5
└── 3×3 MaxPool → 1×1 Conv → 32ch
→ Channel Concat → 256ch (28×28×256)
```

- **1×1畳み込みの役割**: 次元削減（チャンネル数削減）によりパラメータ数を削減
- 学習可能パラメータ: 約500万（VGGの約1/30）

### ResNet

> Kaiming He ら（Microsoft Research）が2015年に発表。ILSVRC 2015優勝（Top-5エラー率3.57%）。

**解決した問題**: 勾配消失問題（`vanishing gradient problem`）

> **勾配消失問題**: ネットワークを深くすると、誤差逆伝播の際に勾配が小さくなりすぎて初期層に届かなくなる問題。56層のネットワークが20層より性能が悪くなることがある。

**残差ブロック（`Residual Block`）**:

$$出力 = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

```
入力 x
  ├── weight layer → ReLU → weight layer → F(x)
  └── identity（スキップ接続）───────────── x
  ↓
F(x) + x → ReLU
```

**残差ブロックの効果**:
1. スキップ接続（`skip connection`）により勾配が直接流れる経路を確保 → 勾配消失を緩和
2. 恒等関数（`identity function`）の学習が可能 → 上位層が下位層より性能が悪化しない

| バージョン | 層数 | パラメータ数 |
|-----------|------|-----------|
| ResNet-50 | 50層 | 約2500万 |
| ResNet-101 | 101層 | 約4500万 |
| ResNet-152 | 152層 | 約6000万 |

### 主要モデルの比較

| モデル | 深さ | イノベーション | パラメータ数 | 備考 |
|--------|------|-------------|-----------|------|
| AlexNet | 8層 | ReLU・Dropout・データ拡張 | ~60M | ディープラーニング普及の火付け役 |
| VGG | 16/19層 | 小フィルタ（3×3）の積み重ね | ~138M | シンプルで理解しやすいが重い |
| Inception | ~22層 | 並列マルチスケール畳み込み | ~5M | 少ないパラメータで高精度 |
| ResNet | 50/101/152層 | 残差ブロック（スキップ接続） | ~25M(ResNet-50) | 超深層ネットワークの学習を可能に |

---

## 4. 転移学習とファインチューニング `Transfer Learning & Fine-Tuning`

### 転移学習の定義

> **転移学習（`Transfer Learning`）**: 事前学習済みモデルが獲得した知識を新しいモデルに転移させる機械学習技術。

| 従来のML | 転移学習 |
|---------|---------|
| 各タスクを独立して学習 | 過去のタスクで学んだ知識を活用 |
| 知識は保持・蓄積されない | 学習が高速・高精度・少データで可能 |

### ディープラーニングにおける転移学習

ディープラーニングでは事前学習済みモデルの**重みを読み込む**ことで実現。

- 2つのモデルは**ヘッド（`head`）以外は同じアーキテクチャ**を共有する
- **ヘッド**: モデルの最終層（タスク固有の予測に使用）
  - 例: ImageNetの1000クラス → 10クラスに変更

### なぜ機能するのか

CNNの早期層（入力に近い層）は汎用的な特徴を学習：
- 低レベル特徴（`low-level features`）: 色・エッジ・テクスチャ
- 中レベル特徴（`mid-level features`）: 形状・パターン
- 高レベル特徴（`high-level features`）: 物体の部位

→ これらは**異なる種類の画像にも適用可能**
→ ヘッドに近い層ほどデータセット固有の特徴を学習

### 転移学習のプロセス

1. 選択したモデルのアーキテクチャをコピー
2. 大規模データセット（例: ImageNet）で学習済みのパラメータをコピー
3. モデルのヘッドを除去
4. 新しいヘッド以前の**全レイヤーを凍結（`freeze`）**
5. 新しいデータセット用の新しいヘッドを追加
6. 新しいデータセットで新しいヘッドのみを訓練

### ファインチューニング `Fine-Tuning`

> **ファインチューニング**: 転移学習の拡張。全事前学習済み層を凍結するのではなく、一部の層も訓練する。

| 手法 | 訓練対象 |
|------|---------|
| 転移学習（`Transfer Learning`） | 新しいヘッドのみ |
| ファインチューニング（`Fine-Tuning`） | 新しいヘッド + 一部の事前学習済み層 |

---

## 5. PyTorchの事前学習済みモデル `Pre-Trained Models in PyTorch`

### 利用可能なモデル

`torchvision.models` から事前学習済みモデルを利用可能：

```
AlexNet, ConvNeXt, DenseNet, EfficientNet, EfficientNetV2,
GoogLeNet, Inception V3, MaxVit, MNASNet, MobileNet V2/V3,
RegNet, ResNet, ResNeXt, ShuffleNet V2, SqueezeNet,
SwinTransformer, VGG, VisionTransformer, Wide ResNet
```

### 事前学習済みモデルの読み込み

```python
# [概念: 事前学習済みモデルの読み込み] 新しいAPI（weights引数推奨）
from torchvision.models import resnet50, ResNet50_Weights

# 事前学習済み重みを使用（推奨方法）
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# 重みなし
resnet50(weights=None)

# Inception V3の読み込み
from torchvision import models
base_model = models.inception_v3(pretrained=True)  # 非推奨、weightsを使用推奨
```

### レイヤーの凍結

```python
# [概念: 全レイヤー凍結] 事前学習済み重みを更新しない
for param in vgg16.parameters():
    param.requires_grad = False

# [概念: 部分凍結] 最初のN層のみ凍結
frozen_layers = 12
for idx, (name, param) in enumerate(vgg16.named_parameters()):
    if idx < frozen_layers:
        param.requires_grad = False
```

### 新しいヘッドの追加

**> Lecture → Lab**: スライドのコード = Colabの Exercise 1/2/3 の `base_model.fc = ...` に対応。

```python
# [概念: ヘッドの置き換え] タスク固有の分類器に変更
num_ftrs = base_model.fc.in_features  # 既存の全結合層の入力次元を取得

base_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 500),   # 中間層（500ユニット）
    nn.ReLU(),
    nn.Linear(500, 10),         # 出力層（クラス数に合わせる）
    # ※ Softmax は不要: nn.CrossEntropyLoss が内部で log_softmax を適用するため
    #   ここで Softmax を付けると二重適用になり収束が悪化する
)
tuned_model = base_model
```

---

## 6. PyTorchコールバック `Callbacks`

### コールバックとは

> **コールバック（`Callback`）**: 訓練ループに追加機能を補足するオプションの拡張機能。Lab 4 では独自の `CustomCallback` クラスを定義して使用する。

| コールバック | 機能 |
|-----------|------|
| `BaseCSVWriter` | 予測出力をCSVファイルに書き込む |
| `GarbageCollector` | 定期的なガベージコレクション |
| `Lambda` | 訓練・評価・予測ループ中に関数を実行 |
| `LearningRateMonitor` | 学習率を記録 |
| `ModuleSummary` | モジュールのサマリーを生成・記録 |
| `PyTorchProfiler` | PyTorch Profilerでプロファイリング |
| `SystemResourcesMonitor` | CPU・GPU・メモリ使用状況を記録 |

### 早期停止 `EarlyStopping`

> **早期停止（`Early Stopping`）**: 監視するメトリクスの改善が止まったとき、自動的に訓練を停止する機能。

**主なパラメータ**:

| パラメータ | 説明 |
|-----------|------|
| `early_stop_patience` | 改善がない場合に待つエポック数 |
| `reduce_lr_factor` | 学習率を下げる乗数 |
| `reduce_lr_patience` | 学習率を下げるまでの待機エポック数 |
| `reduce_lr_min_lr` | 最小学習率 |
| `checkpoint_path` | モデルのチェックポイント保存先 |

**> Lecture → Lab**: Lab 4 では `ignite` 等の外部ライブラリは使わず、PyTorch 標準機能のみで `CustomCallback` を独自実装している。

```python
# [概念: CustomCallback] EarlyStoppingとReduceLROnPlateauを統合した独自クラス
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CustomCallback:
    def __init__(self, early_stop_patience=5, reduce_lr_factor=0.2,
                 reduce_lr_patience=3, reduce_lr_min_lr=1e-7,
                 checkpoint_path='checkpoint.pth'):
        self.early_stop_patience = early_stop_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_min_lr = reduce_lr_min_lr
        self.checkpoint_path = checkpoint_path
        # set_optimizer() / set_model() で後から注入する

# 使用例
custom_callback = CustomCallback()
custom_callback.set_optimizer(optimizer)
custom_callback.set_model(tuned_model)
```

### 学習率スケジューラ `ReduceLROnPlateau`

> メトリクスの改善が止まったとき、自動的に学習率を下げる機能。

`from torch.optim.lr_scheduler import ReduceLROnPlateau`

**主なパラメータ**:

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `mode` | 'min' | 'min': 減少停止で発動、'max': 増加停止で発動 |
| `factor` | 0.1 | 学習率の乗数（new_lr = lr × factor） |
| `patience` | 10 | 改善なしで何エポック待つか |
| `threshold` | 1e-4 | 改善とみなす閾値 |
| `min_lr` | 0 | 最小学習率 |

```python
# [概念: ReduceLROnPlateau] 検証lossが改善しなければ学習率を下げる
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min')

for epoch in range(10):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # validate()の後に呼ぶ
```

**> Lecture → Lab**: Exercise 3 の `CustomCallback` クラス内で `ReduceLROnPlateau` と早期停止を組み合わせて実装。

```python
# [概念: カスタムコールバック] Early StoppingとReduceLROnPlateauの組み合わせ
class CustomCallback:
    def __init__(self, early_stop_patience=5, reduce_lr_factor=0.2, ...):
        self.scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                           factor=reduce_lr_factor, ...)

    def on_epoch_end(self, epoch, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        if self.early_stop_counter >= self.early_stop_patience:
            print("Early stopping triggered!")
            return True  # 訓練を停止

        self.scheduler.step(val_loss)
        return False
```

---

## Lab 4 の概要

| Exercise | タスク | モデル | データセット |
|---------|--------|-------|-----------|
| Exercise 1 | 二値分類 | 転移学習（ResNetベース） | Cats vs Dogs |
| Exercise 2 | 二値分類 | ファインチューニング（VGG16） | Cats vs Dogs |
| Exercise 3 | 多クラス分類（10クラス） | 転移学習+ファインチューニング（Inception_v3） | EuroSAT |

**Exercise 3 の結果（Inception_v3 + EuroSAT）**:

| エポック | 訓練Loss | 訓練精度 | 検証Loss | 検証精度 |
|---------|---------|---------|---------|---------|
| 1 | 2.2145 | 36.8% | 2.0476 | 57.9% |
| 5 | 1.8678 | 64.6% | 1.8251 | 67.9% |
| 10 | 1.7919 | 71.1% | 1.7507 | 74.7% |
| テスト | - | - | 1.7416 | **75.4%** |

**ImageNet正規化の統計値**（転移学習で共通して使用）:
```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
※事前学習時のImageNet統計に合わせて正規化することで転移学習の性能が向上する。

---

## 重要用語まとめ

| 日本語 | 英語 | 説明 |
|--------|------|------|
| データ拡張 | `Data Augmentation` | 既存画像から変換で新画像を生成し訓練データを増やす技術 |
| 過学習 | `overfitting` | 訓練データに特化しすぎて汎化できない状態 |
| 水平反転 | `Horizontal Flipping` | 画像を左右反転 |
| 垂直反転 | `Vertical Flipping` | 画像を上下反転 |
| カラージッター | `Color Jitter` | 輝度・色相などをランダムに変化 |
| 転移学習 | `Transfer Learning` | 事前学習済みモデルの知識を新タスクに転用 |
| ファインチューニング | `Fine-Tuning` | 事前学習済み層の一部も再訓練する転移学習の拡張 |
| ヘッド | `Head` | モデルの最終層（タスク固有の予測部分） |
| 凍結 | `Freeze` | 層のパラメータを更新しないように固定すること |
| 残差ブロック | `Residual Block` | スキップ接続を持つブロック（ResNet） |
| スキップ接続 | `Skip Connection` | 入力をそのまま出力側に加算する接続 |
| 勾配消失問題 | `Vanishing Gradient Problem` | 深いネットワークで勾配が消えて学習できなくなる問題 |
| 早期停止 | `Early Stopping` | 検証性能が改善しなくなったら訓練を自動停止 |
| 学習率スケジューラ | `Learning Rate Scheduler` | 訓練進行に合わせて学習率を動的に調整 |
| ILSVRC | `ImageNet Large Scale Visual Recognition Challenge` | ImageNetを使った大規模画像認識競技会 |

---

## 復習ポイント

- [ ] データ拡張が過学習防止にどう機能するか説明できる
- [ ] `v2.Compose()` で複数の変換をチェーンする書き方を理解している
- [ ] AlexNet・VGG・Inception・ResNetの特徴と違いを説明できる
- [ ] ResNetの残差ブロックが勾配消失問題を解決する仕組みを説明できる
- [ ] 転移学習とファインチューニングの違いを説明できる
- [ ] `requires_grad = False` で層を凍結する実装ができる
- [ ] `ReduceLROnPlateau` と `EarlyStopping` の使いどころを理解している

---

## 混同しやすいポイント

| よく間違えること | 正しい理解 |
|----------------|-----------|
| 転移学習 ＝ ファインチューニング | 転移学習はヘッドのみ訓練。ファインチューニングは事前学習済み層の一部も訓練 |
| データ拡張は推論時にも常に使う | 訓練時のみ使用。推論時は使わない（`model.eval()` で無効化） |
| 凍結 = モデルの削除 | 凍結はパラメータの更新を止めるだけ。順伝播は通常通り行われる |
| Inceptionモジュールは直列 | Inceptionモジュールは並列（複数のフィルタサイズを同時適用して結合） |
| `scheduler.step()` はどこでも呼べる | `ReduceLROnPlateau` は必ず `validate()` の後に呼ぶ |

---

## 理解チェック問題

1. 訓練データが犬の顔画像しかない場合、モデルが全身の犬を認識できない理由を過学習の観点から説明してください。データ拡張でこの問題をどのように緩和できますか？

2. ResNetの残差ブロックで `F(x) + x` という計算をすることで、なぜ勾配消失問題が緩和されるのですか？数式ではなく概念的に説明してください。

3. VGG16でImageNetの分類器を猫vs犬の二値分類器として使いたい場合、転移学習を使うコードの手順を書いてください。（凍結・ヘッド追加・訓練の3ステップで）

<details>
<summary>解答例</summary>

1. 訓練データが正面の顔画像のみだと、モデルはその分布（ポーズ・背景・照明）に特化する（過学習）。データ拡張で水平反転・クロップ・色変化などのバリエーションを生成することで、訓練データの多様性が増し、モデルが実世界の様々な条件に対して汎化しやすくなる。

2. スキップ接続により、誤差逆伝播の際に勾配が `F(x) + x` の `x` の経路をそのまま通り抜けられる（乗算なしで直接流れる）。これにより、途中の層を「ショートカット」して勾配が浅い層にも届くため、消えにくくなる。

3.
```python
# Step 1: 凍結
for param in vgg16.parameters():
    param.requires_grad = False

# Step 2: 新しいヘッドを追加（2クラス）
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, 2)

# Step 3: 新しいヘッドのみ訓練
optimizer = torch.optim.Adam(vgg16.classifier[6].parameters(), lr=0.001)
# 通常の訓練ループへ
```

</details>

---

`#DeepLearning` `#Lecture-4` `#Lab-4` `#DataAugmentation` `#TransferLearning` `#FineTuning` `#ImageNet` `#CNN` `#ResNet` `#VGG` `#Inception` `#AlexNet` `#UTS`
