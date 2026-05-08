# シンチレータ波形位置推定プロジェクト - プロジェクトアーキテクチャ

**プロジェクト名**：シンチレータ通過位置推定（1D CNN ベースアプローチ）
**対象データ**：TOF シンチレーションカウンタ波形（15cm～140cm、23,602 イベント）
**実装言語**：Python 3.10+, PyTorch 2.0+
**実行環境**：Apple M3 MAX (MPS) / CUDA / CPU 対応

---

## 📁 プロジェクト構成

```
Scintillator_Project/
├── dataset/
│   ├── raw/                          # 原始.dat ファイル（15個）
│   │   ├── run7_65.dat
│   │   ├── run8_55.dat
│   │   └── ...
│   └── split/                        # 分割済みデータ（train/val/test）
│       ├── train.json               # 訓練セット（16,514 イベント）
│       ├── val.json                 # 検証セット（3,533 イベント）
│       └── test.json                # テストセット（3,555 イベント）
│
├── src/
│   ├── __init__.py
│   ├── data_parse/
│   │   ├── __init__.py
│   │   ├── dat_file_reader.py        # .dat ファイルパーサ
│   │   ├── split_dataset.py          # データ分割（70/15/15）
│   │   ├── scintillator_dataset.py   # PyTorch Dataset クラス
│   │   └── data_loader.py            # DataLoader 生成
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn1d.py                  # CNN1D モデル（44,257 パラメータ）
│   │
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── train.py                  # 訓練ループ
│   │   ├── evaluate.py               # テスト評価（MAE/RMSE/R²）
│   │   └── evaluate_traditional.py   # 従来手法対比
│   │
│   └── visualize.py                  # 評価結果の可視化
│
├── results/
│   ├── plots/
│   │   ├── 20260504-174051_scatter.png           # 真値 vs 予測値 散布図
│   │   ├── 20260504-174051_error_by_position.png # 位置ごと誤差箱線図
│   │   ├── 20260504-174051_error_histogram.png   # 誤差分布ヒストグラム
│   │   ├── cnn1d.pdf                             # ネットワーク構造図
│   │   └── 波形データを用いた位置推定について報告.pdf  # 最終報告
│   ├── 20260504-130716_best_model.pth            # 最良モデル重み
│   ├── 20260504-130716_test_results.txt          # テスト結果ログ
│   ├── report.md                                 # 中文技術レポート
│   └── report_ja.md                              # 日文技術レポート
│
├── logs/
│   └── 20260504-104005.log           # 訓練ログ（100 epoch 全記録）
│
├── runs/                             # TensorBoard ログディレクトリ
│
├── docs/
│   ├── thesis_chap3_zh.md            # 大場論文第3章の中文翻訳
│   ├── training_log_analysis.md      # 訓練ログ分析
│   └── 1d_cnn_vs_2d_cnn_zh.md        # 1D CNN vs 2D CNN の説明
│
├── pyproject.toml                    # パッケージ設定
└── README.md                         # プロジェクト説明

```

---

## 🔧 セットアップ手順

### 1. 環境構築

```bash
# リポジトリクローン
cd /Users/your_path/GAPS_Project

# Python パッケージをインストール
pip install -e .

# 依存ライブラリ確認
python -c "import torch; print(torch.__version__)"
```

### 2. データ準備

**入力**：`dataset/raw/` に 15 個の `.dat` ファイルを配置

```bash
# データをパース＆ JSON へ変換（不要、すでに split/ に存在）
# 詳細は docs/scintillator_project_guide.md の Phase 1-2 参照
```

**現在の状態**：`dataset/split/` に訓練/検証/テストセットが既に準備完了
- train.json: 16,514 イベント
- val.json: 3,533 イベント
- test.json: 3,555 イベント

---

## 📊 実装ステップ（完了済み）

### Phase 1：データ処理 ✅
- **Step 1-3**：.dat ファイル読み込み → JSON 変換 → データ検証
- **ファイル**：`src/data_parse/dat_file_reader.py`
- **出力**：`dataset/split/{train,val,test}.json`

### Phase 2：データ分割 ✅
- **Step 4-6**：層化分割（70/15/15）、各位置バランス確保
- **ファイル**：`src/data_parse/split_dataset.py`

### Phase 3：Dataset & DataLoader ✅
- **Step 7-8**：PyTorch Dataset クラス実装、波形正規化
- **ファイル**：`src/data_parse/scintillator_dataset.py` + `data_loader.py`
- **入力**：`[batch, 2, 1024]`、正規化済み

### Phase 4：モデル実装 ✅
- **Step 9**：1D CNN 設計
- **ファイル**：`src/models/cnn1d.py`
- **構造**：Conv1d × 3（BN + ReLU + MaxPool）→ AdaptiveAvgPool → FC × 2
- **パラメータ**：44,257

### Phase 5：訓練 & 検証 ✅
- **Step 10-11**：訓練ループ（100 epoch、StepLR scheduler）
- **ファイル**：`src/scripts/train.py`
- **結果**：RMSE = 5.52 cm、R² = 0.984

### Phase 6：テスト評価 ✅
- **Step 12**：テストセット評価
- **ファイル**：`src/scripts/evaluate.py`
- **出力**：`results/{timestamp}_test_results.txt`

### Phase 7：可視化 & レポート ✅
- **Step 13**：3 つの評価図（散布図、箱線図、ヒストグラム）
- **Step 14**：技術レポート作成
- **ファイル**：`src/scripts/visualize.py`
- **出力**：`results/plots/`、`results/report_ja.md`

---

## 🚀 使用方法

### 訓練の実行

```bash
cd /Users/your_path/GAPS_Project/Scintillator_Project
python -m Scintillator_Project.src.scripts.train
```

**出力**：
- `results/{timestamp}_best_model.pth` — 最良モデル
- `logs/{timestamp}.log` — 訓練ログ
- `runs/` — TensorBoard イベント

### TensorBoard で訓練過程を確認

```bash
python -m tensorboard.main --logdir runs
# ブラウザで http://localhost:6006 を開く
```

### テスト評価の実行

```bash
python -m Scintillator_Project.src.scripts.evaluate
```

**出力**：
- `results/{timestamp}_test_results.txt` — 評価指標（MAE/RMSE/R²、位置別誤差）

### 可視化の生成

```bash
python -m Scintillator_Project.src.scripts.visualize
```

**出力**：
- `results/plots/{timestamp}_scatter.png` — 真値 vs 予測値
- `results/plots/{timestamp}_error_by_pos.png` — 位置ごと誤差
- `results/plots/{timestamp}_error_hist.png` — 誤差分布

### 従来手法との対比

```bash
python -m Scintillator_Project.src.scripts.evaluate_traditional
```

**出力**：
```
電荷量比法        | RMSE: 10.87 cm | MAE: 8.29 cm
到達時刻差法     | RMSE: 12.77 cm | MAE: 9.35 cm
組み合わせ法     | RMSE: 9.17 cm  | MAE: 6.73 cm
CNN (本手法)     | RMSE: 5.52 cm  | MAE: 3.35 cm
提升幅度        | 39.8%
```

---

## 📈 主要結果

| 指標 | 値 |
|------|----|
| **テスト RMSE** | 5.52 cm |
| **MAE** | 3.35 cm |
| **R²** | 0.984 |
| **従来法比** | +39.8% 改善 |

### 位置別誤差分析
- **最良位置**：45 cm（RMSE 4.53 cm、N=677）、80 cm（4.83 cm、N=575）
- **最悪位置**：135 cm（RMSE 11.20 cm、N=37）— サンプル不足が原因

---

## 🔍 ファイル説明

### コア実装

| ファイル | 機能 | 入力 | 出力 |
|---------|------|------|------|
| `cnn1d.py` | 1D CNN モデル定義 | [batch, 2, 1024] | [batch, 1] |
| `scintillator_dataset.py` | PyTorch Dataset | JSON | (waveform, label) |
| `data_loader.py` | DataLoader 生成 | split/ JSON | batched tensors |
| `train.py` | 訓練ループ | DataLoader | .pth + .log |
| `evaluate.py` | テスト評価 | .pth + test set | MAE/RMSE/R² |
| `visualize.py` | 結果可視化 | predict + true | PNG グラフ |

### 設定ファイル

| ファイル | 内容 |
|---------|------|
| `pyproject.toml` | パッケージ定義（name: "scintillator-project"） |
| `__init__.py` (各層) | Python パッケージ初期化 |

---

## 📝 使用例：新しいモデルで再訓練

```python
from Scintillator_Project.src.data_parse.data_loader import make_dataloaders
from Scintillator_Project.src.models.cnn1d import CNN1D
import torch

# データロード
split_dir = Path("dataset/split")
train_loader, val_loader, test_loader = make_dataloaders(split_dir, batch_size=64)

# モデル初期化
model = CNN1D()

# 訓練（カスタムループ）
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    for x, y in train_loader:
        pred = model(x)
        loss = criterion(pred, y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## ⚙️ 環境要件

- Python >= 3.10
- PyTorch >= 2.0
- NumPy, Pandas
- Matplotlib, Scikit-learn

**GPU/NPU サポート**：
- CUDA（NVIDIA GPU）
- MPS（Apple Silicon）
- CPU（フォールバック）

---

## 📚 参考資料

- `docs/thesis_chap3_zh.md` — 実験セットアップ詳細（大場論文第3章翻訳）
- `results/report_ja.md` — 完全な技術レポート
- `results/plots/波形データを用いた位置推定について報告.pdf` — 最終報告書

---

## 🤝 保守・拡張

### モデル改善方向
1. **データ拡張**：サンプル少ない位置（135 cm、25 cm など）のオーバーサンプリング
2. **脈冲切割**：有効波形区間の自動抽出（計算量削減）
3. **アンサンブル**：複数モデルの組み合わせ

### 既知の課題
- **サンプル不均衡**：135 cm は N=37 のみ → 誤差 11.20 cm
- **外れ値事象**：15 cm に誤差 > 30 cm の異常サンプル存在
- **伝統手法比較**：本実装は線形回帰、卒論と異なる可能性あり

---

**作成日**：2026-05-05
**最終更新**：2026-05-05
**ステータス**：プロダクション準備完了 ✅
