---
name: generate-study-note
description: This skill should be used when the user asks to "ノートを作って", "ノートを生成して", "スライドからまとめて", "Colabをまとめて", "LectureXのノートを作って", or wants to create a study note from UTS lecture materials.
version: 1.0.0
---

# Skill: 学習ノート生成

UTSの授業資料（スライドPDF・Colabノートブック・Pythonファイル）から学習ノートを生成するスキル。

---

## 入力の確認手順

ノート生成を開始する前に以下を確認する：

1. `data/raw/slides/` にスライドPDFがあるか
2. `data/raw/colab/` に対応するノートブックがあるか
3. `data/raw/python/` にPythonファイルがあるか
4. Lecture番号・Lab番号・トピック名を特定する

入力がない場合はユーザーに確認する。

---

## 生成手順

1. スライドPDFを読み込み、章立てを把握する
2. Colabノートブックを読み込み、対応するコードを把握する
3. `references/note-template.md` のテンプレートと生成ルールに従ってノートを作成する
4. 出力先: `notes/obsidian/LectureX_LabY_トピック名.md`
5. 品質チェックリスト（`references/note-template.md` 末尾）をすべて確認してから保存する

---

## 入力の優先順位

1. **スライド（最優先）** — 講義の本質・概念説明の軸
2. **Colabノートブック** — 実装の詳細・コード例の補足
3. **Pythonファイル** — スクリプトの補足

---

## 言語ルール

- **出力言語**: 日本語メイン
- **英語を必ず併記**: 専門用語、関数名、クラス名、ライブラリ名、モデル名
- **英語のみでよいもの**: コードブロック内のコード
- **曖昧・不明な内容**: 推測せず `> ※ スライドに記載なし。要確認。` と明示する

---

## Additional Resources

- **`references/note-template.md`** — 標準テンプレート全文・各セクション生成ルール・品質チェックリスト
