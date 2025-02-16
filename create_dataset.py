"""
SFT用のデータセットを作成する。

手順:
- public以下のmdファイルを読み込む
- 各ファイルごとに以下の処理を行う
  - yaml形式のメタデータを読み込み、titleを取得
  - メタデータ以降の本文を取得し、Markdown形式でパース
  - 各セクションごとに、学習データを作成する。
- 学習データはjsonl形式で保存する

セクションから学習データを作成する際のルール:
学習データは以下のようなフォーマット
```
## システムプロンプト
あなたは文書検索botです。
入力された文からそれが含まれる文書タイトルとセクションの見出しを返します。

## 検索する文書
{paragraph}

Answer:
## 文書タイトル
{title}

## 見出し
{heading}<|plamo:eos|>
```

記事のパラグラフごとに、上記のフォーマットで学習データを作成すること

使用するライブラリ
- marko
"""

import os
import re
import json
import marko
import marko.md_renderer
import yaml

# New implementation for dataset creation


def create_training_dataset():
    public_dir = os.path.join(os.path.dirname(__file__), "public")
    output_file = os.path.join(os.path.dirname(__file__), "dataset.train.jsonl")
    validate_file = os.path.join(os.path.dirname(__file__), "dataset.validate.jsonl")
    training_examples = []

    renderer = marko.md_renderer.MarkdownRenderer()

    for filename in os.listdir(public_dir):
        if not filename.endswith(".md"):
            continue
        file_path = os.path.join(public_dir, filename)
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Extract YAML front-matter
        m = re.match(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
        if m:
            yaml_text, body = m.groups()
            meta = yaml.safe_load(yaml_text)
            title = meta.get("title", filename)
        else:
            body = content
            title = filename

        # Parse the Markdown body using marko
        parsed = marko.parse(body)
        current_heading = ""  # holds the most recent heading text
        for element in parsed.children:
            if element.__class__.__name__ == "Heading":
                current_heading = renderer.render(element).strip()
                # Remove Markdown heading markers (e.g., ##)
                current_heading = re.sub(r"^#+\s*", "", current_heading)
            elif element.__class__.__name__ == "Paragraph":
                paragraph = renderer.render(element).strip()[:100]
                if not paragraph:
                    continue
                # Use the last seen heading for the dataset's "見出し"
                training_text = f"""## システムプロンプト
あなたは文書検索botです。
入力された文からそれが含まれる文書タイトルとセクションの見出しを返します。

## 検索する文書
{paragraph}

## Answer:
#### 文書タイトル
{title}

#### 見出し
{current_heading}<|plamo:eos|>"""
                training_examples.append({"text": training_text})

    # split the training examples into training and validation sets
    # ランダムにシャッフルして、9:1で分割
    training_examples = sorted(training_examples, key=lambda x: hash(x["text"]))
    split_index = int(len(training_examples) * 0.9)
    training_examples, validation_examples = (
        training_examples[:split_index],
        training_examples[split_index:],
    )

    with open(output_file, "w", encoding="utf-8") as out:
        for example in training_examples:
            out.write(json.dumps(example, ensure_ascii=False) + "\n")

    with open(validate_file, "w", encoding="utf-8") as out:
        for example in validation_examples:
            out.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    create_training_dataset()
    # ...existing code...
