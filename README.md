# arXiv Text Summarization: Comparative Analysis

This project implements and evaluates five different text summarization approaches on arXiv research paper abstracts, comparing their performance using six complementary evaluation metrics.

## Overview

The project generates single-sentence extractive summaries from arXiv paper abstracts and evaluates them against article titles using multiple similarity metrics. This enables a comprehensive comparison of traditional NLP techniques versus modern transformer-based approaches.

## Summarization Methods

1. **Frequency-Based** - Selects sentences containing the most frequent meaningful words
2. **TF-IDF** - Identifies sentences with the highest term frequency-inverse document frequency scores
3. **LexRank** - Graph-based algorithm using sentence similarity and centrality
4. **TextRank** - PageRank-inspired approach for sentence importance
5. **T5 Transformer** - Pre-trained sequence-to-sequence model for abstractive summarization

## Evaluation Metrics

Each summary is evaluated using six metrics:

- **FuzzyWuzzy Score** - Token-based string similarity
- **Modified Jaccard Similarity** - Ratio of common tokens to total unique tokens
- **FastText Cosine Similarity** - Semantic similarity using word embeddings
- **ROUGE-1** - Unigram overlap
- **ROUGE-2** - Bigram overlap
- **ROUGE-L** - Longest common subsequence

## Project Structure

```
project/
├── data/
│   └── arxiv-metadata-oai-snapshot.json    # Dataset (not included)
├── main.py                                  # Main implementation
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
├── results_frequency.csv                    # result files:
├── results_tfidf.csv
├── results_lexrank.csv
├── results_textrank.csv
├── results_t5.csv
├── comprehensive_statistics.csv
├── comparison_plots.png
└── overall_comparison.png
```

## Installation

### Install Dependencies

use the requirements file:

```bash
pip install -r requirements.txt
```

### Download Dataset

1. Download the arXiv dataset from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
2. Place `arxiv-metadata-oai-snapshot.json` in a `data/` directory

## Usage

Run the main script:

```bash
python main.py
```

The script will:
1. Load 100 arXiv abstracts (configurable)
2. Generate summaries using all five approaches
3. Evaluate each summary with six metrics
4. Save individual results for each approach
5. Perform comprehensive statistical analysis
6. Generate comparison visualizations

**Note**: First execution will download NLTK data (~50MB), FastText embeddings (~1GB), and T5 model (~250MB).

## Configuration

To process a different number of articles, modify the following line in `main.py`:

```python
abstracts, titles = load_arxiv_data(
    'data/arxiv-metadata-oai-snapshot.json', n_samples=100)
```

Change `n_samples=100` to your desired value.

For GPU acceleration with T5, change:

```python
device=-1  # CPU
```

to:

```python
device=0  # GPU
```

## Output Files

### Individual Approach Results
- `results_frequency.csv` - Frequency-based method results with all metrics
- `results_tfidf.csv` - TF-IDF method results
- `results_lexrank.csv` - LexRank method results
- `results_textrank.csv` - TextRank method results
- `results_t5.csv` - T5 transformer method results

Each file contains:
- Article ID, title, abstract
- Generated summary
- All six evaluation scores

### Analysis Files
- `comprehensive_statistics.csv` - Statistical summary (mean, std dev, min, max) for all approaches
- `comparison_plots.png` - Six-panel visualization comparing approaches across metrics
- `overall_comparison.png` - Overall performance comparison

## Expected Performance

ROUGE-L scores typically range from 0.20 to 0.40 for this task, which is normal given:
- Technical nature of arXiv abstracts
- Domain-specific terminology
- Article titles are not comprehensive summaries
- Single-sentence constraint

Expected performance ranking:
1. T5 Transformer (highest semantic understanding)
2. LexRank/TextRank (balanced graph-based approaches)
3. TF-IDF (good distinctive term identification)
4. Frequency-based (competitive baseline)

## Troubleshooting

### Missing NLTK Resources

If you encounter NLTK resource errors:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### Memory Issues

Reduce the number of samples:

```python
n_samples=50  # or smaller
```

### Slow T5 Processing

- Expected processing time: 10-20 minutes for 100 articles on CPU
- Use GPU acceleration if available
- Reduce sample size for testing

### Import Errors

Ensure all dependencies are installed:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Implementation Details

### Evaluation Suite

The `EvaluationSuite` class provides a unified interface for all metrics:

```python
eval_suite = EvaluationSuite()
scores = eval_suite.evaluate(summary, title)
# Returns: {fuzzy_score, jaccard_score, fasttext_score, 
#           rouge1_score, rouge2_score, rougeL_score}
```

### Summarization Approaches

All summarizers follow a consistent interface:

```python
summarizer = SummarizationApproaches()
summary = summarizer.method_name(abstract)
```

### Analysis Function

The `analyze_results()` function:
- Loads all result files
- Computes comprehensive statistics
- Generates visualizations
- Identifies best-performing approaches per metric

## Limitations

- **Extractive constraint**: Methods primarily select existing sentences (except T5)
- **Single sentence**: Significant context loss compared to multi-sentence summaries
- **Domain specificity**: Performance varies with scientific domain
- **Evaluation bias**: ROUGE favors lexical overlap over semantic similarity
- **No fine-tuning**: T5 not adapted to scientific abstracts


## References

- **Dataset**: [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- **LexRank**: Erkan & Radev (2004)
- **TextRank**: Mihalcea & Tarau (2004)
- **ROUGE**: Lin (2004)
- **T5**: Raffel et al. (2020)
- **Sumy Library**: [Documentation](https://github.com/miso-belica/sumy)

## License

MIT License

## Author

Hasaan Ahmed, Aqib Ilyas and Minhal Shafiq (2025)

## Acknowledgments

- Cornell University for the arXiv dataset
- Hugging Face for transformer implementations
- Contributors to NLTK, Gensim, and Sumy libraries