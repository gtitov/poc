import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup paths
MODEL_NAME = "./model" if Path("./model").exists() else "TrustSafeAI/RADAR-Vicuna-7B"

print(f"Loading model {MODEL_NAME}")

try:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_NAME), local_files_only=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_NAME), local_files_only=False
    )
    model.eval()
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")
    sys.exit(1)


def preprocess_text(text):
    """Removes YAML frontmatter and HTML tags."""
    # 1. Remove YAML frontmatter (content between first two '---' markers)
    # The regex looks for '---' at the very start, then anything until the next '---'
    text = re.sub(r"^\s*---\s*.*?\s*---\s*", "", text, flags=re.DOTALL)

    # 2. Remove HTML tags (e.g., <br>, <div class="...">)
    text = re.sub(r"<[^>]+>", "", text)

    return text.strip()


def analyze_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Apply preprocessing
        text = preprocess_text(raw_text)

        if len(text) < 10:
            print(f"Skipping {filepath}: text too short after cleaning.")
            return None

        # Process in the first 512 tokens
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ai_prob = probs[0][1].item()
            return ai_prob

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def main():
    files = sys.argv[1:]
    if not files:
        print("No files provided.")
        return

    results = []
    for file in files:
        score = analyze_file(file)
        if score is not None:
            results.append(f"| `{Path(file).name}` | {score:.2%} |")
            print(f"Analyzed: {file} -> {score:.2%}")
        else:
            results.append(f"| `{file}` | Error/Skip |")

    with open("comment.md", "w", encoding="utf-8") as f:
        f.write("| Файл | Вероятность генерации текста ИИ |\n| :--- | :--- |\n")
        f.write("\n".join(results))
        f.write(
            "\n\n---\n\n"
            "Оценка выполнена моделью [TrustSafeAI/RADAR-Vicuna-7B [↗]](https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B) ([arXiv:2307.03838](https://arxiv.org/abs/2307.03838))"
        )


if __name__ == "__main__":
    main()
