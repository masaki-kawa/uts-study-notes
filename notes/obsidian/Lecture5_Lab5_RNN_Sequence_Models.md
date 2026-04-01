# Lecture 5: シーケンスモデルと時系列予測 `RNN & Sequence Models`

> **対応**: Lecture 5 ↔ Lab 5 | スライド: `Deep Learning - Lecture 5.pptx.pdf` | Colab: `Deep_Learning_Lab5_Exercise1_Solutions.ipynb`, `Deep_Learning_Lab5_Exercise2_Solutions.ipynb`

---

## 目次

1. [[#1. シーケンスデータとは]]
2. [[#2. 時系列分析 (TSA)]]
3. [[#3. 評価指標]]
4. [[#4. RNN アーキテクチャ]]
5. [[#5. 勾配消失・勾配爆発問題]]
6. [[#6. LSTM]]
7. [[#7. RNN 出力タイプ]]
8. [[#8. PyTorch API: nn.RNN / nn.LSTM]]
9. [[#9. データ前処理パターン]]
10. [[#10. Lab 5 演習まとめ]]
11. [[#重要用語まとめ]]
12. [[#復習ポイント]]
13. [[#混同しやすいポイント]]
14. [[#理解チェック問題]]

---

## 1. シーケンスデータとは

> シーケンスデータ `sequence data` とは、**順序に意味がある**データのこと。前後の要素が互いに関係を持つ。

### 4 種類のシーケンスデータ

| 種類 | 例 |
|------|----|
| テキスト `Text` | 文章、翻訳、感情分析 |
| 音声 `Audio` | 音声認識、音楽生成 |
| 動画 `Video` | 動作認識、動画キャプション |
| 時系列 `Time Series` | 株価、気温、センサデータ |

### ニューラルネットワークの種類比較

| モデル | 入力 | 特徴 | 用途例 |
|--------|------|------|--------|
| 順伝播 NN `Feed-Forward NN` | 固定長ベクトル | 前後関係を扱えない | 画像分類 |
| CNN `Convolutional NN` | グリッド状データ | 局所パターン検出 | 画像認識 |
| シーケンスモデル `Sequence Models` | 可変長シーケンス | 時間的依存関係を保持 | 翻訳・予測 |

---

## 2. 時系列分析 (TSA)

> 時系列分析 `Time Series Analysis (TSA)` とは、時間に沿って変化するデータのパターンを分析し、**過去の観測から未来を予測**すること。

### 時系列の 4 成分

| 成分 | 説明 |
|------|------|
| トレンド `Trend` | 長期的な増減傾向 |
| 季節性 `Seasonality` | 一定周期で繰り返すパターン（年・週・日単位） |
| 周期性 `Cyclical` | 不規則な長期的波動（景気サイクルなど） |
| 不規則成分 `Irregular` | ランダムなノイズ |

### 予測モデル

#### ARIMA
> ARIMA = **AR**（自己回帰）+ **I**（和分・差分）+ **MA**（移動平均）
> 統計的アプローチで非定常時系列にも対応する。

| 成分 | 英語 | 役割 |
|------|------|------|
| AR | Autoregressive | 過去の値を説明変数として使う |
| I | Integrated (Differencing) | 差分をとって定常化する |
| MA | Moving Average | 過去の予測誤差を使う |

#### 指数平滑化 `Exponential Smoothing`
- 直近のデータに大きな重みを置き、古いデータの重みを指数的に小さくする
- パラメータ α（平滑化係数）で過去への依存度を調整

### 応用例

| タスク | データ | 手法 |
|--------|--------|------|
| 株価予測 `Stock Price Prediction` | Netflix 株価（終値） | LSTM |
| 気象予測 `Weather Forecasting` | 気温・湿度・風速 | RNN |

---

## 3. 評価指標

時系列・回帰タスクで使う主要な評価指標 `evaluation metrics`。

| 指標 | 数式 | 特徴 |
|------|------|------|
| MAE（平均絶対誤差）`Mean Absolute Error` | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | 外れ値に頑健。誤差の方向を無視 |
| MSE（平均二乗誤差）`Mean Squared Error` | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | 大きな誤差を強調。微分可能で損失関数に使いやすい |
| RMSE（二乗平均平方根誤差）`Root Mean Squared Error` | $\sqrt{\text{MSE}}$ | 元の単位と同じスケールで解釈しやすい |

> **Lecture → Lab**: スライドの MAE/MSE/RMSE = Colab の `mean_absolute_error`, `mean_squared_error`（sklearn）+ `math.sqrt()` に対応。

---

## 4. RNN アーキテクチャ

> 再帰型ニューラルネットワーク `Recurrent Neural Network (RNN)` は、隠れ状態 `hidden state` を通じて**過去の情報を次のステップに引き継ぐ**アーキテクチャ。

### 基本構造

各タイムステップ t において：

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
o_t = W_hy · h_t + b_y
```

| 記号 | 意味 |
|------|------|
| `h_t` | 時刻 t の隠れ状態（hidden state） |
| `h_{t-1}` | 直前の隠れ状態（過去の記憶） |
| `x_t` | 時刻 t の入力 |
| `W_hh`, `W_xh`, `W_hy` | 重み行列（全タイムステップで共有） |
| `b_h`, `b_y` | バイアス |

### BPTT（時間を通じた逆伝播）

> 誤差逆伝播法 `Backpropagation Through Time (BPTT)` は、時間軸を展開した RNN において時刻をさかのぼりながら勾配を計算する手法。

### シーケンスモデルの設計基準

1. **可変長入力** `variable-length sequences` への対応
2. **長期依存** `long-term dependencies` の保持
3. **順序** `order` の考慮
4. **パラメータ共有** `parameter sharing`（各タイムステップで同一の重みを使用）

---

## 5. 勾配消失・勾配爆発問題

### 問題の概要

| 問題 | 内容 | 結果 |
|------|------|------|
| 勾配消失 `Vanishing Gradient` | BPTT で勾配が指数的に縮小 | 遠い過去の情報が伝わらない（短期記憶のみ） |
| 勾配爆発 `Exploding Gradient` | 勾配が指数的に増大 | 重みが発散して学習が不安定になる |

### 解決策

#### 勾配爆発への対策

| 手法 | 説明 |
|------|------|
| 単位行列初期化 `Identity Matrix Init` | 重みを単位行列で初期化し、勾配の急拡大を防ぐ |
| 切り捨て BPTT `Truncated BPTT` | 一定のタイムステップ数で逆伝播を打ち切る |
| 勾配クリッピング `Gradient Clipping` | 勾配のノルムが閾値を超えたらスケール縮小 |

#### 勾配消失への対策

| 手法 | 説明 |
|------|------|
| 重み初期化 `Weight Initialization` | Xavier/He 初期化など |
| 活性化関数の選択 `Activation Function` | ReLU 系を使用 |
| **LSTM** | ゲート機構で長期記憶を維持（最も効果的） |

> **Lecture → Lab**: 勾配クリッピング = Exercise 1 の `train_one_epoch` 内で `max_norm=2` として実装。

```python
# [概念: 勾配クリッピング] ノルムが max_norm=2 を超えたらスケールを縮小
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
optimizer.step()
```

---

## 6. LSTM

> 長短期記憶 `Long Short-Term Memory (LSTM)` は、**セル状態 `cell state`** と**3つのゲート**によって、長期的な依存関係を効果的に学習するアーキテクチャ。

### RNN との違い

| | RNN | LSTM |
|-|-----|------|
| 記憶機構 | 隠れ状態 h_t のみ | セル状態 C_t（長期）+ 隠れ状態 h_t（短期）|
| 勾配消失 | 起きやすい | ゲートで軽減 |
| 計算コスト | 低い | 高い（ゲートが多い分）|

### LSTM の 3 つのゲート

| ゲート | 活性化関数 | 役割 |
|--------|-----------|------|
| 忘却ゲート `Forget Gate` | sigmoid | セル状態から何を捨てるか決定（0=全忘却, 1=全保持）|
| 入力ゲート `Input Gate` | sigmoid × tanh | 新しい情報をどれだけ追加するか決定 |
| 出力ゲート `Output Gate` | sigmoid × tanh(C_t) | 何を次の隠れ状態として出力するか決定 |

### セル状態の更新

```
C_t = f_t * C_{t-1}  +  i_t * g_t
         ↑                 ↑
    （忘却: 過去を消す）（入力: 新情報を追加）

h_t = o_t * tanh(C_t)
```

> **Lecture → Lab**: LSTM アーキテクチャ = Exercise 2 の `class LSTM` が実装。`nn.LSTM` + `nn.Linear` の組み合わせで終値を予測。

---

## 7. RNN 出力タイプ

RNN は入出力の組み合わせにより 5 つのモードで使用できる。

| タイプ | 入力 | 出力 | 用途例 |
|--------|------|------|--------|
| One-to-One | 1 | 1 | 通常の NN（シーケンスなし） |
| One-to-Many | 1 | シーケンス | 画像キャプション生成 |
| Many-to-One | シーケンス | 1 | 感情分析、時系列予測 |
| Many-to-Many（半同期）| シーケンス | シーケンス（遅延あり）| 機械翻訳（Encoder-Decoder）|
| Many-to-Many（全同期）| シーケンス | シーケンス（同時）| 動画クラス分類 |

> 今回の Lab 5（気温予測・株価予測）は **Many-to-One** パターン：過去 N ステップのシーケンスから 1 つの値を予測。

---

## 8. PyTorch API: nn.RNN / nn.LSTM

### nn.RNN パラメータ

```python
# [概念: RNN 層] 時系列データを処理する再帰型層
nn.RNN(
    input_size,      # 各タイムステップの入力特徴量数
    hidden_size,     # 隠れ状態のサイズ（ユニット数）
    num_layers,      # 積み重ね層数（デフォルト: 1）
    nonlinearity,    # 活性化関数: 'tanh'（デフォルト）または 'relu'
    bias,            # バイアス項を使うか（デフォルト: True）
    batch_first,     # True → 入力形状 (batch, seq, feature)
    dropout          # 層間ドロップアウト率（num_layers > 1 のとき有効）
)
```

### nn.LSTM パラメータ

```python
# [概念: LSTM 層] ゲート機構を持つ長短期記憶層
nn.LSTM(
    input_size,      # 各タイムステップの入力特徴量数
    hidden_size,     # 隠れ状態とセル状態のサイズ
    num_layers,      # 積み重ね層数
    batch_first,     # True → 入力形状 (batch, seq, feature)
    dropout          # 層間ドロップアウト率
)
# 出力: (output, (h_n, c_n)) ← h_n=隠れ状態, c_n=セル状態
```

### RNN vs LSTM コード比較

| 項目 | RNN (Exercise 1) | LSTM (Exercise 2) |
|------|-----------------|-------------------|
| 層 | `nn.RNN(...)` | `nn.LSTM(...)` |
| BatchNorm | あり (`nn.BatchNorm1d`) | なし |
| Dropout | あり (`dropout_rate=0.4`) | なし |
| hidden_size | 64 | 128 |
| num_layers | 2 | 1 |
| Grad Clipping | あり (`max_norm=2`) | なし |
| num_epochs | 30 | 100 |

---

## 9. データ前処理パターン

時系列データの一般的な前処理フロー。

### ステップ 1: 日付処理

```python
# [概念: インデックス化] 時系列データは日付をインデックスにして管理
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
```

### ステップ 2: カテゴリ変数のエンコーディング

```python
# [概念: ラベルエンコーディング] カテゴリ列を数値に変換
weather_mapping = {w: idx for idx, w in enumerate(data['weather'].unique())}
data['weather'] = data['weather'].map(weather_mapping)
```

### ステップ 3: ルックバックウィンドウ作成

```python
# [概念: ルックバック] 過去 lookback_size ステップ分の特徴量を列として展開
lookback_size = 24
shifted_columns = {f'temperature(t-{i})': data['temperature'].shift(i)
                   for i in range(1, lookback_size + 1)}
```

### ステップ 4: 学習・検証・テスト分割

```python
# [概念: 時系列分割] 時系列はシャッフルせずに時間順に分割する
train_size = int(data.shape[0] * 0.7)   # 70%
val_size   = int(data.shape[0] * 0.15)  # 15%
# 残り 15% がテスト
```

### ステップ 5: 正規化

```python
# [概念: MinMaxScaler] 訓練データで fit し、検証・テストには transform のみ適用
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_df[:] = scaler.fit_transform(train_df)  # fit + transform
val_df[:]   = scaler.transform(val_df)         # transform のみ
test_df[:]  = scaler.transform(test_df)        # transform のみ
```

> **重要**: 検証・テストデータで `fit_transform` を使うと**データリーク `data leakage`** が発生する。必ず訓練データのみで `fit` すること。

### ステップ 6: シーケンス生成

```python
# [概念: シーケンス化] 固定長の入力系列と対応するターゲットを生成
def create_sequences(data, seq_length, target_index=0):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])          # 過去 seq_length ステップ
        y.append(data[i+seq_length, target_index])  # 次のステップの目標値
    return np.array(X), np.array(y)
```

---

## 10. Lab 5 演習まとめ

### Exercise 1: 気象気温予測（RNN）

| 項目 | 内容 |
|------|------|
| データセット | 気象データ（8 列: 気温・露点・湿度・風速・視界・気圧・天気）|
| 予測対象 | 次の 1 時間の気温 `temperature` |
| シーケンス長 | 24（過去 24 時間）|
| モデル | RNN + BatchNorm + Dropout + Linear |
| パラメータ | hidden_size=64, num_layers=2, dropout=0.4 |
| 最適化 | Adam, lr=0.001, weight_decay=1e-5 |
| エポック数 | 30, バッチサイズ 64 |
| 勾配クリッピング | max_norm=2 |

```python
# [概念: RNN モデル定義] BatchNorm と Dropout で正則化
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout_rate=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]          # 最後のタイムステップのみ使用（Many-to-One）
        out = self.batch_norm(out)
        out = self.dropout(out)
        return self.fc(out)
```

### Exercise 2: 株価（終値）予測（LSTM）

| 項目 | 内容 |
|------|------|
| データセット | Netflix 株価（7 列: Date, Open, High, Low, Close, Adj Close, Volume）|
| 予測対象 | 終値 `Close` (target_index=3) |
| シーケンス長 | 7（過去 7 日間）|
| モデル | LSTM + Linear |
| パラメータ | input_size=12, hidden_size=128, num_layers=1 |
| 最適化 | Adam, lr=0.001, weight_decay=1e-5 |
| エポック数 | 100, バッチサイズ 64 |

```python
# [概念: LSTM モデル定義] セル状態で長期記憶を保持
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = out[:, -1, :]          # 最後のタイムステップ（Many-to-One）
        return self.fc(out)
```

### Lab 5 比較まとめ

| | Exercise 1（気温予測）| Exercise 2（株価予測）|
|-|---------------------|---------------------|
| データ特性 | 気象・多変量 | 金融・多変量 |
| モデル | RNN | LSTM |
| seq_length | 24 時間 | 7 日 |
| 正則化 | BatchNorm + Dropout + GradClip | Weight Decay のみ |
| epochs | 30 | 100 |
| 結果可視化 | 実際 vs 予測 気温グラフ | 実際 vs 予測 終値グラフ |

---

## 重要用語まとめ

| 日本語 | 英語 | 説明 |
|--------|------|------|
| シーケンスデータ | `Sequence Data` | 順序に意味があるデータ（テキスト・音声・時系列など）|
| 時系列分析 | `Time Series Analysis (TSA)` | 過去の時間的パターンから未来を予測する手法 |
| 自己回帰モデル | `ARIMA` | AR+I+MA の組み合わせで非定常時系列を扱う統計モデル |
| 指数平滑化 | `Exponential Smoothing` | 過去データに指数的な重みを付けて予測する手法 |
| 平均絶対誤差 | `MAE` | 予測誤差の絶対値の平均 |
| 平均二乗誤差 | `MSE` | 予測誤差の二乗の平均。損失関数としてよく使われる |
| 二乗平均平方根誤差 | `RMSE` | MSE の平方根。元のスケールで誤差を解釈できる |
| 再帰型ニューラルネットワーク | `RNN` | 隠れ状態で過去の情報を引き継ぐアーキテクチャ |
| 隠れ状態 | `Hidden State` | RNN が過去の情報を保持するベクトル |
| 時間方向逆伝播 | `BPTT (Backpropagation Through Time)` | RNN の学習のため時間軸を展開して勾配を計算する手法 |
| 勾配消失 | `Vanishing Gradient` | 長いシーケンスで勾配が指数的に小さくなる問題 |
| 勾配爆発 | `Exploding Gradient` | 長いシーケンスで勾配が指数的に大きくなる問題 |
| 勾配クリッピング | `Gradient Clipping` | 勾配ノルムに上限を設けて爆発を防ぐ手法 |
| 長短期記憶 | `LSTM (Long Short-Term Memory)` | ゲート機構でRNNの勾配消失問題を解決したアーキテクチャ |
| セル状態 | `Cell State` | LSTM の長期記憶を保持するベクトル |
| 忘却ゲート | `Forget Gate` | LSTM がどの情報を捨てるかを制御するゲート |
| 入力ゲート | `Input Gate` | LSTM がどの新情報を追加するかを制御するゲート |
| 出力ゲート | `Output Gate` | LSTM が何を次の隠れ状態として出力するかを制御するゲート |
| ルックバック | `Lookback Window` | 予測に使う過去のステップ数 |
| 正規化 | `MinMaxScaler` | 特徴量を [0, 1] のスケールに変換する前処理 |
| データリーク | `Data Leakage` | 検証・テストデータの情報が学習に混入する問題 |

---

## 復習ポイント

- [ ] RNN の隠れ状態の更新式 `h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b_h)` を理解する
- [ ] 勾配消失と勾配爆発の違い・原因・それぞれの解決策を説明できる
- [ ] LSTM の 3 つのゲート（忘却・入力・出力）の役割を説明できる
- [ ] 時系列データで MinMaxScaler を適用するとき、訓練データでのみ `fit` する理由を説明できる
- [ ] MAE / MSE / RMSE の違いと使い分けを説明できる
- [ ] `nn.RNN` と `nn.LSTM` のパラメータ（input_size, hidden_size, num_layers）を使いこなせる
- [ ] Many-to-One パターンで最後のタイムステップ `out[:, -1, :]` を使う理由を説明できる

---

## 混同しやすいポイント

| よく間違えること | 正しい理解 |
|----------------|-----------|
| `fit_transform` を検証・テストにも使う | 検証・テストには `transform` のみ。`fit` は訓練データ専用 |
| RNN と LSTM の出力形式が同じだと思う | LSTM は `(output, (h_n, c_n))` を返す。RNN は `(output, h_n)` のみ |
| 勾配消失 = 勾配爆発の解決策が同じ | 別の問題。勾配爆発→クリッピング、勾配消失→LSTM・初期化変更 |
| セル状態と隠れ状態が同じと思う | セル状態 `c_n`（長期記憶）と隠れ状態 `h_n`（短期記憶）は別物 |
| ARIMA は深層学習の手法だと思う | ARIMA は統計的手法。RNN/LSTM が深層学習ベースのシーケンスモデル |
| `batch_first=False`（デフォルト）のまま使う | デフォルトは `(seq, batch, feature)`。Lab では `batch_first=True` で `(batch, seq, feature)` に変更 |

---

## 理解チェック問題

1. RNN が通常のフィードフォワード NN と比べて「シーケンスデータ」に適している理由を、**隠れ状態**の仕組みを使って説明しなさい。

2. LSTM の「セル状態」がなぜ勾配消失問題の解決に効果的なのかを、**忘却ゲートと入力ゲートの動作**を含めて説明しなさい。

3. 以下のコードにはデータリーク `data leakage` の問題がある。どこが問題で、どのように修正すべきか答えなさい。
   ```python
   scaler = MinMaxScaler()
   train_df[:] = scaler.fit_transform(train_df)
   val_df[:]   = scaler.fit_transform(val_df)   # ← 問題箇所
   test_df[:]  = scaler.fit_transform(test_df)  # ← 問題箇所
   ```

4. 株価予測（Exercise 2）で `seq_length=7` を設定した場合、`create_sequences` 関数が生成する `X` の形状は何になるか。データ数を N として答えなさい。

<details>
<summary>解答例</summary>

1. 通常のフィードフォワード NN は各入力を独立して処理するが、RNN は隠れ状態 `h_t` を通じて前のタイムステップの情報を保持し、次の入力と組み合わせて処理する。これにより、シーケンス内の順序と文脈を考慮した予測が可能になる。

2. セル状態 `C_t` は「情報の高速道路」として機能し、長距離にわたって情報が直接流れる。忘却ゲート（sigmoid）が不要な情報を 0〜1 の係数でフィルタリングし、入力ゲートが新しい情報を加算する。この加算的な更新により、乗算の連鎖が少なくなり、勾配が消失しにくくなる。

3. 問題: `val_df` と `test_df` に `fit_transform` を使っている。これにより検証・テストデータの統計情報がスケーラーに混入し、データリークが発生する。
   修正: `val_df[:] = scaler.transform(val_df)` および `test_df[:] = scaler.transform(test_df)` に変更する（`fit` は訓練データのみ）。

4. `X` の形状は `(N - seq_length, 7, input_features)` になる。具体的には `(N-7, 7, 12)` （seq_length=7, input_size=12 の場合）。

</details>

---

`#DeepLearning` `#Lecture-5` `#Lab-5` `#RNN` `#LSTM` `#TimeSeries` `#UTS`
