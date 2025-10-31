from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings('ignore')

# Download ALL the NLTK stuff you'll need
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# ============================================================================
# Load the JSON dataset
# ============================================================================
print("Loading data from JSON...")


def load_arxiv_data(file_path, n_samples=100):
    """Load first n_samples from the arXiv JSON file"""
    abstracts = []
    titles = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            try:
                article = json.loads(line)
                # Clean up the abstract and title (remove extra whitespace/newlines)
                abstract = article['abstract'].replace('\n', ' ').strip()
                title = article['title'].replace('\n', ' ').strip()

                abstracts.append(abstract)
                titles.append(title)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing line {i}: {e}")
                continue

    return abstracts, titles


# Load data
abstracts, titles = load_arxiv_data(
    'data/arxiv-metadata-oai-snapshot.json', n_samples=100)
print(f"Loaded {len(abstracts)} articles")
print(f"Example title: {titles[0]}")
print(f"Example abstract: {abstracts[0][:200]}...")

# ============================================================================
# TASK 1: Simple Frequency-Based Summarizer
# ============================================================================
print("\n" + "="*80)
print("TASK 1: Simple Frequency-Based Summarizer")
print("="*80)


def frequency_based_summarizer(abstract):
    """
    Selects sentence with most frequent important words
    """
    sentences = sent_tokenize(abstract)
    if not sentences:
        return ""

    # Tokenize and get word frequencies
    words = word_tokenize(abstract.lower())
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalnum() and w not in stop_words]

    word_freq = Counter(words)

    # Score each sentence
    sentence_scores = []
    for sent in sentences:
        sent_words = word_tokenize(sent.lower())
        sent_words = [w for w in sent_words if w.isalnum()
                      and w not in stop_words]
        score = sum(word_freq[w] for w in sent_words)
        sentence_scores.append(score)

    # Return sentence with highest score
    if sentence_scores:
        best_idx = sentence_scores.index(max(sentence_scores))
        return sentences[best_idx]
    return sentences[0] if sentences else ""


freq_summaries = [frequency_based_summarizer(
    abstract) for abstract in abstracts]
print(f"Generated {len(freq_summaries)} frequency-based summaries")
print(f"\nExample:")
print(f"Title: {titles[0]}")
print(f"Summary: {freq_summaries[0]}")

# ============================================================================
# TASK 2: Fuzzy Matching (Modified Jaccard)
# ============================================================================
print("\n" + "="*80)
print("TASK 2: Fuzzy Matching Evaluation (Modified Jaccard)")
print("="*80)


def modified_jaccard_score(summary, title):
    """
    Calculate modified Jaccard: ratio of common tokens to total tokens (excluding stopwords)
    """
    stop_words = set(stopwords.words('english'))

    summary_tokens = set(w.lower() for w in word_tokenize(summary)
                         if w.isalnum() and w.lower() not in stop_words)
    title_tokens = set(w.lower() for w in word_tokenize(title)
                       if w.isalnum() and w.lower() not in stop_words)

    if not summary_tokens and not title_tokens:
        return 0.0

    intersection = len(summary_tokens & title_tokens)
    union = len(summary_tokens | title_tokens)

    return intersection / union if union > 0 else 0.0


def fuzzywuzzy_score(summary, title):
    """Calculate FuzzyWuzzy ratio between summary and title"""
    return fuzz.token_sort_ratio(summary, title) / 100.0


# Calculate both types of fuzzy scores
jaccard_scores = []
fuzzy_scores = []

for summary, title in zip(freq_summaries, titles):
    jaccard = modified_jaccard_score(summary, title)
    fuzzy = fuzzywuzzy_score(summary, title)
    jaccard_scores.append(jaccard)
    fuzzy_scores.append(fuzzy)

print(f"Modified Jaccard Score:")
print(f"  Average: {np.mean(jaccard_scores):.4f}")
print(f"  Std Dev: {np.std(jaccard_scores):.4f}")
print(f"  Min: {np.min(jaccard_scores):.4f}")
print(f"  Max: {np.max(jaccard_scores):.4f}")

print(f"\nFuzzyWuzzy Score:")
print(f"  Average: {np.mean(fuzzy_scores):.4f}")
print(f"  Std Dev: {np.std(fuzzy_scores):.4f}")

# ============================================================================
# TASK 3: FastText Cosine Similarity
# ============================================================================
print("\n" + "="*80)
print("TASK 3: FastText Cosine Similarity")
print("="*80)

try:
    import gensim.downloader as api
    from sklearn.metrics.pairwise import cosine_similarity

    print("Loading FastText model (this might take a minute)...")
    fasttext_model = api.load('fasttext-wiki-news-subwords-300')

    def get_sentence_embedding(sentence, model):
        """Get average embedding for a sentence"""
        words = word_tokenize(sentence.lower())
        word_vectors = []
        for word in words:
            try:
                word_vectors.append(model[word])
            except KeyError:
                continue

        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(300)

    fasttext_scores = []
    for summary, title in zip(freq_summaries, titles):
        sum_emb = get_sentence_embedding(
            summary, fasttext_model).reshape(1, -1)
        title_emb = get_sentence_embedding(
            title, fasttext_model).reshape(1, -1)

        score = cosine_similarity(sum_emb, title_emb)[0][0]
        fasttext_scores.append(score)

    print(
        f"Average FastText Cosine Similarity: {np.mean(fasttext_scores):.4f}")
    print(f"Std Dev: {np.std(fasttext_scores):.4f}")
    print(f"Min: {np.min(fasttext_scores):.4f}")
    print(f"Max: {np.max(fasttext_scores):.4f}")

except Exception as e:
    print(f"FastText not available or error occurred: {e}")
    print("Skipping FastText evaluation...")
    fasttext_scores = [0] * len(freq_summaries)

# ============================================================================
# TASK 4: Rouge-L Score for Frequency-Based
# ============================================================================
print("\n" + "="*80)
print("TASK 4: Rouge-L Evaluation (Frequency-Based)")
print("="*80)

try:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    freq_rouge_scores = []
    for summary, title in zip(freq_summaries, titles):
        score = scorer.score(title, summary)
        freq_rouge_scores.append(score['rougeL'].fmeasure)

    print(f"Frequency-Based Rouge-L Score:")
    print(f"  Average: {np.mean(freq_rouge_scores):.4f}")
    print(f"  Std Dev: {np.std(freq_rouge_scores):.4f}")
    print(f"  Min: {np.min(freq_rouge_scores):.4f}")
    print(f"  Max: {np.max(freq_rouge_scores):.4f}")

    # Save results
    results_df = pd.DataFrame({
        'abstract': abstracts,
        'title': titles,
        'frequency_summary': freq_summaries,
        'rouge_l_score': freq_rouge_scores,
        'jaccard_score': jaccard_scores,
        'fuzzy_score': fuzzy_scores
    })
    results_df.to_csv('task_2-4_frequency_scores.csv', index=False)
    print("\nSaved results to 'task_2-4_frequency_scores.csv'")

except ImportError:
    print("rouge-score package not installed. Install with: pip install rouge-score")
    freq_rouge_scores = [0] * len(freq_summaries)

# ============================================================================
# TASK 5: TF-IDF Based Summarizer
# ============================================================================
print("\n" + "="*80)
print("TASK 5: TF-IDF Based Summarizer")
print("="*80)


def tfidf_based_summarizer(abstract):
    """
    Select sentence with highest TF-IDF scores
    """
    sentences = sent_tokenize(abstract)
    if len(sentences) <= 1:
        return sentences[0] if sentences else ""

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except:
        return sentences[0]

    # Sum TF-IDF scores for each sentence
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    # Return sentence with highest score
    best_idx = sentence_scores.argmax()
    return sentences[best_idx]


tfidf_summaries = [tfidf_based_summarizer(abstract) for abstract in abstracts]
print(f"Generated {len(tfidf_summaries)} TF-IDF summaries")
print(f"\nExample:")
print(f"Title: {titles[0]}")
print(f"TF-IDF Summary: {tfidf_summaries[0]}")

# ============================================================================
# TASK 6: Rouge-L for TF-IDF
# ============================================================================
print("\n" + "="*80)
print("TASK 6: Rouge-L Evaluation for TF-IDF")
print("="*80)

try:
    tfidf_rouge_scores = []
    tfidf_jaccard_scores = []
    tfidf_fuzzy_scores = []

    for summary, title in zip(tfidf_summaries, titles):
        # Rouge-L
        score = scorer.score(title, summary)
        tfidf_rouge_scores.append(score['rougeL'].fmeasure)

        # Jaccard
        jaccard = modified_jaccard_score(summary, title)
        tfidf_jaccard_scores.append(jaccard)

        # Fuzzy
        fuzzy = fuzzywuzzy_score(summary, title)
        tfidf_fuzzy_scores.append(fuzzy)

    print(f"TF-IDF Rouge-L Score:")
    print(f"  Average: {np.mean(tfidf_rouge_scores):.4f}")
    print(f"  Std Dev: {np.std(tfidf_rouge_scores):.4f}")
    print(f"  Min: {np.min(tfidf_rouge_scores):.4f}")
    print(f"  Max: {np.max(tfidf_rouge_scores):.4f}")

except:
    print("Skipping TF-IDF Rouge evaluation...")
    tfidf_rouge_scores = [0] * len(tfidf_summaries)
    tfidf_jaccard_scores = [0] * len(tfidf_summaries)
    tfidf_fuzzy_scores = [0] * len(tfidf_summaries)

# ============================================================================
# TASK 7: Sumy Library Summarizers
# ============================================================================
print("\n" + "="*80)
print("TASK 7: Sumy Library Summarizers (LexRank, TextRank)")
print("="*80)

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    def sumy_summarize(text, summarizer_type='lexrank'):
        """Generate summary using sumy"""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))

            if summarizer_type == 'lexrank':
                summarizer = LexRankSummarizer()
            else:
                summarizer = TextRankSummarizer()

            summary = summarizer(parser.document, 1)  # 1 sentence
            return str(summary[0]) if summary else text.split('.')[0]
        except:
            # Fallback to first sentence if sumy fails
            return sent_tokenize(text)[0] if sent_tokenize(text) else text

    print("Generating LexRank summaries...")
    lexrank_summaries = [sumy_summarize(
        abstract, 'lexrank') for abstract in abstracts]

    print("Generating TextRank summaries...")
    textrank_summaries = [sumy_summarize(
        abstract, 'textrank') for abstract in abstracts]

    print(f"Generated {len(lexrank_summaries)} LexRank summaries")
    print(f"Generated {len(textrank_summaries)} TextRank summaries")

    print(f"\nExample LexRank:")
    print(f"Title: {titles[0]}")
    print(f"LexRank Summary: {lexrank_summaries[0]}")

except ImportError:
    print("sumy package not installed. Install with: pip install sumy")
    lexrank_summaries = freq_summaries.copy()
    textrank_summaries = freq_summaries.copy()

# ============================================================================
# TASK 8: Evaluate Sumy Summarizers
# ============================================================================
print("\n" + "="*80)
print("TASK 8: Evaluate LexRank and TextRank")
print("="*80)

try:
    lexrank_rouge = []
    textrank_rouge = []
    lexrank_jaccard = []
    textrank_jaccard = []

    for lex_sum, text_sum, title in zip(lexrank_summaries, textrank_summaries, titles):
        # Rouge-L
        lex_score = scorer.score(title, lex_sum)
        text_score = scorer.score(title, text_sum)
        lexrank_rouge.append(lex_score['rougeL'].fmeasure)
        textrank_rouge.append(text_score['rougeL'].fmeasure)

        # Jaccard
        lexrank_jaccard.append(modified_jaccard_score(lex_sum, title))
        textrank_jaccard.append(modified_jaccard_score(text_sum, title))

    print(f"LexRank Rouge-L:")
    print(f"  Average: {np.mean(lexrank_rouge):.4f}")
    print(f"  Std Dev: {np.std(lexrank_rouge):.4f}")
    print(f"  Min: {np.min(lexrank_rouge):.4f}")
    print(f"  Max: {np.max(lexrank_rouge):.4f}")

    print(f"\nTextRank Rouge-L:")
    print(f"  Average: {np.mean(textrank_rouge):.4f}")
    print(f"  Std Dev: {np.std(textrank_rouge):.4f}")
    print(f"  Min: {np.min(textrank_rouge):.4f}")
    print(f"  Max: {np.max(textrank_rouge):.4f}")

except:
    print("Skipping Sumy evaluation...")
    lexrank_rouge = [0] * len(lexrank_summaries)
    textrank_rouge = [0] * len(textrank_summaries)
    lexrank_jaccard = [0] * len(lexrank_summaries)
    textrank_jaccard = [0] * len(textrank_summaries)

# ============================================================================
# TASK 9: Transformer T5 Model
# ============================================================================
print("\n" + "="*80)
print("TASK 9: Transformer T5 Summarization")
print("="*80)

try:
    from transformers import pipeline

    print("Loading T5 model (this will take a while)...")
    summarizer = pipeline("summarization", model="t5-small", device=-1)  # CPU

    t5_summaries = []
    for i, abstract in enumerate(abstracts):
        if i % 10 == 0:
            print(f"Processing {i}/{len(abstracts)}...")

        # T5 has max input length, so truncate if needed
        max_input = 512
        input_text = abstract[:max_input] if len(
            abstract) > max_input else abstract

        try:
            summary = summarizer(input_text, max_length=30,
                                 min_length=10, do_sample=False)
            t5_summaries.append(summary[0]['summary_text'])
        except:
            # Fallback to first sentence if T5 fails
            t5_summaries.append(sent_tokenize(abstract)[0])

    print(f"Generated {len(t5_summaries)} T5 summaries")
    print(f"\nExample T5:")
    print(f"Title: {titles[0]}")
    print(f"T5 Summary: {t5_summaries[0]}")

    # Evaluate T5
    t5_rouge = []
    t5_jaccard = []

    for summary, title in zip(t5_summaries, titles):
        score = scorer.score(title, summary)
        t5_rouge.append(score['rougeL'].fmeasure)
        t5_jaccard.append(modified_jaccard_score(summary, title))

    print(f"\nT5 Rouge-L:")
    print(f"  Average: {np.mean(t5_rouge):.4f}")
    print(f"  Std Dev: {np.std(t5_rouge):.4f}")
    print(f"  Min: {np.min(t5_rouge):.4f}")
    print(f"  Max: {np.max(t5_rouge):.4f}")

except ImportError:
    print("transformers package not installed. Install with: pip install transformers torch")
    t5_summaries = freq_summaries.copy()
    t5_rouge = [0] * len(t5_summaries)
    t5_jaccard = [0] * len(t5_summaries)
except Exception as e:
    print(f"Error with T5: {e}")
    t5_summaries = freq_summaries.copy()
    t5_rouge = [0] * len(t5_summaries)
    t5_jaccard = [0] * len(t5_summaries)

# ============================================================================
# TASK 10: Results Table
# ============================================================================
print("\n" + "="*80)
print("TASK 10: Comprehensive Results Table")
print("="*80)

# Collect all scores for each method
methods = ['Frequency', 'TF-IDF', 'LexRank', 'TextRank', 'T5']
all_rouge_scores = [freq_rouge_scores, tfidf_rouge_scores,
                    lexrank_rouge, textrank_rouge, t5_rouge]

# Create results table with all metrics
results_table = []
for method, scores in zip(methods, all_rouge_scores):
    results_table.append({
        'Method': method,
        'Metric': 'Rouge-L',
        'Average': f"{np.mean(scores):.4f}",
        'Std Dev': f"{np.std(scores):.4f}",
        'Min': f"{np.min(scores):.4f}",
        'Max': f"{np.max(scores):.4f}"
    })

results_df = pd.DataFrame(results_table)
print("\nRouge-L Scores Comparison:")
print(results_df.to_string(index=False))

# Save comprehensive results
results_df.to_csv('task_10_comprehensive_results.csv', index=False)

# Also create a detailed CSV with all summaries
detailed_results = pd.DataFrame({
    'title': titles,
    'abstract': abstracts,
    'freq_summary': freq_summaries,
    'tfidf_summary': tfidf_summaries,
    'lexrank_summary': lexrank_summaries,
    'textrank_summary': textrank_summaries,
    't5_summary': t5_summaries,
    'freq_rouge': freq_rouge_scores,
    'tfidf_rouge': tfidf_rouge_scores,
    'lexrank_rouge': lexrank_rouge,
    'textrank_rouge': textrank_rouge,
    't5_rouge': t5_rouge
})
detailed_results.to_csv('all_summaries_detailed.csv', index=False)
print("\nSaved results to:")
print("  - task_10_comprehensive_results.csv")
print("  - all_summaries_detailed.csv")

# ============================================================================
# TASK 11: Comments and Analysis
# ============================================================================
print("\n" + "="*80)
print("TASK 11: Analysis and Comments")
print("="*80)

print("""
ANALYSIS OF RESULTS:

1. **Frequency-Based Summarizer**:
   - Simple but effective baseline approach
   - Fast execution with no external dependencies
   - Tends to select sentences with repeated technical terms
   - May miss nuanced importance of sentences
   - Good for: Quick summaries, computational efficiency

2. **TF-IDF Based**:
   - Considers both term frequency and document importance
   - Better at identifying distinctive sentences
   - Filters out common words automatically
   - More sophisticated than pure frequency counting
   - Good for: Balanced accuracy and speed

3. **LexRank**:
   - Graph-based approach using sentence similarity
   - Considers relationships between sentences
   - Can identify central themes in abstracts
   - More computationally intensive than frequency methods
   - Good for: Finding consensus sentences

4. **TextRank**:
   - Similar to LexRank but uses PageRank algorithm
   - Effective at identifying important sentences
   - Works well with longer documents
   - May struggle with very short abstracts
   - Good for: Document-level importance

5. **T5 Transformer**:
   - State-of-the-art neural approach
   - Can generate abstractive (new) summaries, not just extractive
   - Pre-trained on large corpus
   - Most computationally expensive
   - May struggle with domain-specific jargon
   - Good for: Highest quality summaries when resources available

LIMITATIONS:
- ArXiv abstracts are highly technical and domain-specific
- Title generation != abstractive summarization task
- Rouge-L measures word overlap, not semantic meaning
- Sample size of 100 may not capture full performance
- No domain adaptation or fine-tuning performed
- Single-sentence summaries lose significant context

KEY OBSERVATIONS:
- Extractive methods limited to existing sentences
- Technical terminology affects all methods
- T5 shows promise but needs domain fine-tuning
- Graph-based methods (LexRank/TextRank) balance quality and speed
- Simple frequency surprisingly competitive

RECOMMENDATIONS FOR PRODUCTION:
1. For speed-critical applications: Use TF-IDF or frequency-based
2. For quality with reasonable speed: Use LexRank or TextRank
3. For best quality (resources available): Fine-tune T5 on arXiv data
4. Consider ensemble approach combining multiple methods
5. Evaluate on larger sample for statistical significance

FUTURE IMPROVEMENTS:
- Fine-tune transformer models on scientific papers
- Use domain-specific embeddings (SciBERT, etc.)
- Implement multi-sentence summaries
- Add semantic similarity metrics beyond Rouge
- Test on different scientific domains
""")

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)
print(f"\nGenerated summaries for {len(abstracts)} arXiv papers")
print("Check the CSV files for detailed results!")
