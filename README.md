# Movie Review Sentiment Analyzer

A sentiment analysis app that classifies movie reviews as positive or negative using a fine-tuned DistilBERT transformer.

## Live Demo
- [HuggingFace Spaces Demo](https://huggingface.co/spaces/Nikhil-Vaishnav-17/distilbert-imdb-demo)
- [HuggingFace Model](https://huggingface.co/Nikhil-Vaishnav-17/distilbert-imdb-sentiment)

## About the Project

This project fine-tunes DistilBERT on the IMDB dataset (25,000 training reviews) to classify movie reviews as positive or negative. It also compares the transformer approach against a BiLSTM model trained from scratch on the same dataset, showing the accuracy and efficiency tradeoffs between the two approaches.

## Model Comparison

| Model | Dataset | Accuracy | Inference Time | Model Size |
|-------|---------|----------|----------------|------------|
| BiLSTM (from scratch) | IMDB (25k) | 87% | <100ms (CPU) | ~2.5MB |
| DistilBERT (3k subset) | IMDB (3k) | 83.4% | 6.33ms (GPU) | ~268MB |
| DistilBERT (full) | IMDB (25k) | 90.9% | 14.17ms (GPU) | ~268MB |

## Key Finding

DistilBERT trained on the full 25k dataset outperforms the BiLSTM by ~4% accuracy despite requiring no manual preprocessing — the transformer's built-in tokenizer handles that automatically. Even more notably, DistilBERT trained on only 3,000 examples came within 4% of the BiLSTM trained on 25,000, demonstrating how pretraining on millions of sentences gives it a head start on understanding language context.

The tradeoff is model size — DistilBERT is approximately 100x larger than the BiLSTM (~268MB vs ~2.5MB), which makes it less suitable for memory-constrained environments like mobile or edge devices.

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Nikhil-vaishnav-17/distilbert-classifier
cd distilbert-classifier

# Create and activate virtual environment
uv venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
uv pip install -r requirements.txt
# or if pyproject.toml is present
uv sync

# Run the app
python app.py
```

Then open your browser at `http://localhost:7860`

## Project Structure

```
distilbert-classifier/
├── 01_finetune_distilbert.ipynb   # Full training notebook with explanations
├── app.py                          # Gradio inference app
├── pyproject.toml                  # Project dependencies and metadata
├── requirements.txt                # List of dependencies
└── README.md                       # Project documentation
```

## Tech Stack

- PyTorch
- HuggingFace Transformers
- Gradio
- IMDB Dataset (via HuggingFace Datasets)