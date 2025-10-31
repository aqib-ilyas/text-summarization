# arXiv Text Summarization Project

Yo, this project implements and compares 5 different text summarization methods on arXiv research paper abstracts. Because apparently one method wasn't enough.

## What This Does

Takes arXiv paper abstracts and generates single-sentence summaries using:

-   Frequency-based (the OG simple approach)
-   TF-IDF (slightly less simple)
-   LexRank (graph-based, fancy)
-   TextRank (also graph-based, slightly different flavor)
-   T5 Transformer (the heavy hitter)

Then compares them all using Rouge-L, Jaccard similarity, and FuzzyWuzzy scores.

## Project Structure

```
project/
├── data/
│   └── arxiv-metadata-oai-snapshot.json    # Your dataset (not included, too big)
├── summarization.py                         # Main code
├── README.md                                # You're reading it
└── output files (generated after running):
    ├── task_2-4_frequency_scores.csv
    ├── task_10_comprehensive_results.csv
    └── all_summaries_detailed.csv
```

## Setup

### 1. Install Python Dependencies

```bash
pip install pandas numpy nltk fuzzywuzzy python-Levenshtein gensim scikit-learn rouge-score sumy transformers torch
```

Or if you're lazy (recommended):

```bash
pip install -r requirements.txt
```

### 2. Get the Dataset

Download the arXiv dataset from: https://www.kaggle.com/datasets/Cornell-University/arxiv/data

Put the `arxiv-metadata-oai-snapshot.json` file in a `data/` folder.

### 3. Run It

```bash
python summarization.py
```

Grab a coffee. The T5 model takes forever to download and run. First time running will also download NLTK data and FastText embeddings.

## What Each Task Does

### Task 1: Frequency-Based Summarizer

Counts word frequencies, picks the sentence with most important words. Simple but surprisingly effective.

### Task 2: Fuzzy Matching

Compares generated summaries to titles using:

-   Modified Jaccard similarity (set intersection/union)
-   FuzzyWuzzy token matching

### Task 3: FastText Embeddings

Uses pre-trained word embeddings to calculate cosine similarity between summary and title. More semantic than just word matching.

### Task 4: Rouge-L Evaluation

Standard summarization metric. Measures longest common subsequence.

### Task 5: TF-IDF Summarizer

Term Frequency-Inverse Document Frequency. Picks sentences with distinctive vocabulary.

### Task 6: Evaluate TF-IDF

Same metrics as Task 4 but for TF-IDF summaries.

### Task 7: Sumy Library

Uses pre-built implementations of LexRank and TextRank. Because why reinvent the wheel?

### Task 8: Evaluate Sumy

Metrics for LexRank and TextRank summaries.

### Task 9: T5 Transformer

The big boy. Uses Google's T5 model for abstractive summarization. Can generate new sentences instead of just extracting existing ones.

### Task 10: Results Table

Compares all methods side-by-side. Shows average, std dev, min, max for each.

### Task 11: Analysis

Written commentary on results and limitations. Required because the professor wants to see if you actually understand what you're doing.

## Output Files

### `task_2-4_frequency_scores.csv`

Detailed results for frequency-based method including all metrics.

### `task_10_comprehensive_results.csv`

Summary table comparing all 5 methods.

### `all_summaries_detailed.csv`

Every summary from every method plus their scores. The mother lode.

## Expected Results

Don't expect miracles. arXiv abstracts are technical AF and titles are often cryptic. Your Rouge-L scores will probably be in the 0.2-0.4 range. That's normal.

**Best performers** (usually):

1. T5 - if it doesn't choke on the jargon
2. LexRank/TextRank - good balance
3. TF-IDF - solid middle ground
4. Frequency - surprisingly competitive
5. Your patience - completely destroyed

## Troubleshooting

### "Resource punkt_tab not found"

The code downloads it automatically. If it still fails, manually run:

```python
import nltk
nltk.download('punkt_tab')
```

### "No module named 'transformers'"

```bash
pip install transformers torch
```

### T5 is slow AF

Yeah. It is. That's transformers for you. Run on GPU if you can (change `device=-1` to `device=0` in the code). Otherwise go make a sandwich.

### "FileNotFoundError: data/arxiv-metadata-oai-snapshot.json"

You didn't download the dataset. See Setup step 2.

### Out of memory

The script only loads 100 papers by default. If even that's too much, change `n_samples=100` to something smaller in the code.

## Customization

Want more papers? Change this line in `summarization.py`:

```python
abstracts, titles = load_arxiv_data('data/arxiv-metadata-oai-snapshot.json', n_samples=100)
```

Change `n_samples=100` to whatever you want. But be warned: T5 will take FOREVER with more samples.

Want different T5 settings? Modify this:

```python
summary = summarizer(input_text, max_length=30, min_length=10, do_sample=False)
```

## Requirements

-   Python 3.7+
-   4GB+ RAM (8GB recommended for T5)
-   Internet connection (first run downloads models)
-   Patience (seriously)

## Notes

-   First run downloads ~1GB of models (FastText + T5)
-   Processing 100 papers takes 10-20 minutes depending on your hardware
-   Rouge-L scores between 0.2-0.4 are normal for this task
-   The code handles errors gracefully and will skip methods if dependencies fail

## License

MIT or whatever. It's a class project. Do what you want with it.

## Author

Someone who had to do this NLP project and now you're benefiting from it. You're welcome.

---
