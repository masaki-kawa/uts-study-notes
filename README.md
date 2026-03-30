# UTS Study Notes

UTSの授業資料（英語スライド・Google Colab・Pythonファイル）をもとに、日本語の復習ノートを生成するプロジェクト。

## 目的

- 授業内容を構造化された日本語ノートとして整理する
- 重要な専門用語・関数名は英語も併記する
- Obsidian で活用できる Markdown 形式で出力する

## 入力

| 種別 | パス | 内容 |
|------|------|------|
| スライド | `data/raw/slides/` | 授業スライド (PDF 等) |
| Colab | `data/raw/colab/` | Google Colab ノートブック (.ipynb) |
| Python | `data/raw/python/` | Pythonファイル (.py) |

## 出力

| 種別 | パス | 内容 |
|------|------|------|
| 学習ノート | `notes/obsidian/` | Obsidian 用 Markdown |
| 処理済みデータ | `data/processed/` | 中間生成物 |

## ディレクトリ構成

```
uts-study-notes/
├── data/
│   ├── raw/
│   │   ├── slides/
│   │   ├── colab/
│   │   └── python/
│   └── processed/
├── notes/
│   └── obsidian/
├── prompts/        # Claude / Codex 向けプロンプトテンプレート
├── scripts/        # 自動化スクリプト
└── docs/           # 補足ドキュメント
```

## エージェント役割分担

| エージェント | 担当 |
|-------------|------|
| Claude Code | Planning, repository understanding, review, task breakdown |
| Codex | Implementation, bug fixing, testing, focused coding tasks |
