# NLU-A2

Clear instructions to run Problem 1 and Problem 2 are provided below.

## 1) Prerequisites

- OS: Windows/Linux/macOS
- Python: 3.10+ recommended
- Internet required for:
	- Problem 1 web scraping
	- NLTK downloads (first run)
	- Problem 2 Task 0 (Anthropic API)

## 2) One-time setup

From project root:

```bash
python -m venv .venv
```

Activate environment:

- Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

- Linux/macOS:

```bash
source .venv/bin/activate
```

Install required packages:

```bash
pip install requests beautifulsoup4 nltk pdfplumber matplotlib wordcloud numpy gensim scikit-learn torch anthropic
```

## 3) Problem 1: IITJ corpus + Word2Vec + semantics + visualization

Folder: `Problem1`

```bash
cd Problem1
```

### Recommended run order

1. Data collection and corpus creation (web + PDF scraping):

```bash
python problem1_data_collection.py
```

This creates/updates:

- `iitj_data.json`
- `corpus.txt`
- `wordcloud.png`

2. Corpus statistics:

```bash
python corpus_stats.py
```

3. Train Word2Vec models (from scratch + Gensim, multiple hyperparameters):

```bash
python Problem2_train_word2vec.py
```

This creates:

- `models/` (many `.npy` and `.model` files)
- `training_results.csv`

4. Semantic analysis (nearest neighbors + analogies):

```bash
python Problem3_semantic_analysis.py
```

5. 2D embedding visualizations (PCA + t-SNE):

```bash
python Problem4_visualization.py
```

This creates image files like:

- `pca_*.png`
- `tsne_*.png`

### Optional script

Train only one 300-dim scratch CBOW model:

```bash
python train_300dim.py
```

### Important note for Tasks 3 and 4

`Problem2_train_word2vec.py` uses `MIN_COUNT = 5`.
In `Problem3_semantic_analysis.py` and `Problem4_visualization.py`, keep `MIN_COUNT` consistent with Task 2 for correct vocabulary-index alignment.

## 4) Problem 2: Character-level Indian name generation

Folder: `Problem2`

```bash
cd ../Problem2
```

### Option A (recommended): use existing dataset file

`TrainingNames.txt` is already present. You can skip Task 0.

Run:

1. Train all three models and generate outputs:

```bash
python task1_train_models.py
```

This creates:

- `saved_models/vanilla_rnn.pt`
- `saved_models/blstm.pt`
- `saved_models/rnn_attention.pt`
- `generated_vanilla_rnn.txt`
- `generated_blstm.txt`
- `generated_rnn_attention.txt`

2. Quantitative evaluation (novelty + diversity):

```bash
python task2_evaluation.py
```

3. Qualitative analysis:

```bash
python task3_qualitative.py
```

4. Model parameter count and size (Vanilla RNN):

```bash
python model_info.py
```

### Option B: regenerate dataset using Anthropic API

If you want to recreate `TrainingNames.txt` from API:

1. Set API key (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY="your_api_key_here"
```

2. Run Task 0:

```bash
python task0_generate_names.py
```

Then continue with Task 1 -> Task 2 -> Task 3.

## 5) Quick command summary

From project root:

```bash
cd Problem1
python problem1_data_collection.py
python corpus_stats.py
python Problem2_train_word2vec.py
python Problem3_semantic_analysis.py
python Problem4_visualization.py

cd ../Problem2
python task1_train_models.py
python task2_evaluation.py
python task3_qualitative.py
python model_info.py
```

## 6) Troubleshooting

- If NLTK tokenizer errors occur, rerun `problem1_data_collection.py` once with internet; it auto-downloads required NLTK resources.
- If Gensim install fails on old Python versions, use Python 3.10+.
- If Torch uses CPU only, training will still run but slower.
- If Task 0 fails, verify `ANTHROPIC_API_KEY` is set in the same terminal session.