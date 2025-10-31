from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from fuzzywuzzy import fuzz
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
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
# EVALUATION SUITE (Tasks 2-4)
# ============================================================================
print("\n" + "="*80)
print("SETTING UP EVALUATION SUITE")
print("="*80)


class EvaluationSuite:
    """
    Evaluation suite that calculates 6 metrics:
    1. FuzzyWuzzy score
    2. Modified Jaccard score
    3. FastText cosine similarity
    4. Rouge-1 F-measure
    5. Rouge-2 F-measure
    6. Rouge-L F-measure
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Load FastText model
        try:
            import gensim.downloader as api
            print("Loading FastText model (this might take a minute)...")
            self.fasttext_model = api.load('fasttext-wiki-news-subwords-300')
            self.fasttext_available = True
        except Exception as e:
            print(f"FastText not available: {e}")
            self.fasttext_available = False
        
        # Load Rouge scorer
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.rouge_available = True
        except ImportError:
            print("rouge-score package not installed. Install with: pip install rouge-score")
            self.rouge_available = False

    def modified_jaccard_score(self, summary, title):
        """Calculate modified Jaccard: ratio of common tokens to total tokens"""
        summary_tokens = set(w.lower() for w in word_tokenize(summary)
                            if w.isalnum() and w.lower() not in self.stop_words)
        title_tokens = set(w.lower() for w in word_tokenize(title)
                          if w.isalnum() and w.lower() not in self.stop_words)

        if not summary_tokens and not title_tokens:
            return 0.0

        intersection = len(summary_tokens & title_tokens)
        union = len(summary_tokens | title_tokens)

        return intersection / union if union > 0 else 0.0

    def fuzzywuzzy_score(self, summary, title):
        """Calculate FuzzyWuzzy ratio between summary and title"""
        return fuzz.token_sort_ratio(summary, title) / 100.0

    def get_sentence_embedding(self, sentence):
        """Get average embedding for a sentence using FastText"""
        if not self.fasttext_available:
            return np.zeros(300)
        
        words = word_tokenize(sentence.lower())
        word_vectors = []
        for word in words:
            try:
                word_vectors.append(self.fasttext_model[word])
            except KeyError:
                continue

        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(300)

    def fasttext_cosine_score(self, summary, title):
        """Calculate FastText cosine similarity"""
        if not self.fasttext_available:
            return 0.0
        
        sum_emb = self.get_sentence_embedding(summary).reshape(1, -1)
        title_emb = self.get_sentence_embedding(title).reshape(1, -1)
        
        score = cosine_similarity(sum_emb, title_emb)[0][0]
        return score

    def rouge_scores(self, summary, title):
        """Calculate Rouge-1, Rouge-2, and Rouge-L F-measures"""
        if not self.rouge_available:
            return 0.0, 0.0, 0.0
        
        scores = self.rouge_scorer.score(title, summary)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure
        
        return rouge1, rouge2, rougeL

    def evaluate(self, summary, title):
        """
        Evaluate a summary against a title.
        Returns dict with all 6 scores.
        """
        fuzzy = self.fuzzywuzzy_score(summary, title)
        jaccard = self.modified_jaccard_score(summary, title)
        fasttext = self.fasttext_cosine_score(summary, title)
        rouge1, rouge2, rougeL = self.rouge_scores(summary, title)
        
        return {
            'fuzzy_score': fuzzy,
            'jaccard_score': jaccard,
            'fasttext_score': fasttext,
            'rouge1_score': rouge1,
            'rouge2_score': rouge2,
            'rougeL_score': rougeL
        }


# Initialize evaluation suite
eval_suite = EvaluationSuite()
print("Evaluation suite initialized!")

# ============================================================================
# SUMMARIZATION APPROACHES
# ============================================================================
print("\n" + "="*80)
print("DEFINING SUMMARIZATION APPROACHES")
print("="*80)


class SummarizationApproaches:
    """Collection of different summarization methods"""

    @staticmethod
    def frequency_based_summarizer(abstract):
        """Selects sentence with most frequent important words"""
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

    @staticmethod
    def tfidf_based_summarizer(abstract):
        """Select sentence with highest TF-IDF scores"""
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

    @staticmethod
    def lexrank_summarizer(abstract):
        """Generate summary using LexRank"""
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lex_rank import LexRankSummarizer

            parser = PlaintextParser.from_string(abstract, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            summary = summarizer(parser.document, 1)  # 1 sentence
            return str(summary[0]) if summary else sent_tokenize(abstract)[0]
        except:
            # Fallback to first sentence if sumy fails
            sentences = sent_tokenize(abstract)
            return sentences[0] if sentences else abstract

    @staticmethod
    def textrank_summarizer(abstract):
        """Generate summary using TextRank"""
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer

            parser = PlaintextParser.from_string(abstract, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, 1)  # 1 sentence
            return str(summary[0]) if summary else sent_tokenize(abstract)[0]
        except:
            # Fallback to first sentence
            sentences = sent_tokenize(abstract)
            return sentences[0] if sentences else abstract

    @staticmethod
    def t5_summarizer(abstract):
        """Generate summary using T5 transformer"""
        try:
            from transformers import pipeline
            
            # Initialize summarizer (cache it for efficiency)
            if not hasattr(SummarizationApproaches, '_t5_pipeline'):
                print("Loading T5 model (first time only)...")
                SummarizationApproaches._t5_pipeline = pipeline(
                    "summarization", model="t5-small", device=-1)
            
            summarizer = SummarizationApproaches._t5_pipeline
            
            # T5 has max input length, so truncate if needed
            max_input = 512
            input_text = abstract[:max_input] if len(abstract) > max_input else abstract
            
            summary = summarizer(input_text, max_length=30,
                               min_length=10, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"T5 error: {e}")
            # Fallback to first sentence
            sentences = sent_tokenize(abstract)
            return sentences[0] if sentences else abstract


# Initialize summarization approaches
summarizers = SummarizationApproaches()
print("Summarization approaches defined!")

# ============================================================================
# GENERATE SUMMARIES AND EVALUATE EACH APPROACH
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUMMARIES AND EVALUATING APPROACHES")
print("="*80)

approaches = {
    'frequency': summarizers.frequency_based_summarizer,
    'tfidf': summarizers.tfidf_based_summarizer,
    'lexrank': summarizers.lexrank_summarizer,
    'textrank': summarizers.textrank_summarizer,
    't5': summarizers.t5_summarizer
}

# Process each approach
for approach_name, summarizer_func in approaches.items():
    print(f"\n{'='*60}")
    print(f"Processing: {approach_name.upper()}")
    print(f"{'='*60}")
    
    # Generate summaries
    summaries = []
    results = []
    
    for i, (abstract, title) in enumerate(zip(abstracts, titles)):
        if i % 20 == 0:
            print(f"  Processing article {i}/{len(abstracts)}...")
        
        # Generate summary
        summary = summarizer_func(abstract)
        summaries.append(summary)
        
        # Evaluate summary
        scores = eval_suite.evaluate(summary, title)
        
        # Store results
        result_row = {
            'article_id': i,
            'title': title,
            'abstract': abstract,
            'summary': summary,
            **scores  # Unpack all scores
        }
        results.append(result_row)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    filename = f'results_{approach_name}.csv'
    results_df.to_csv(filename, index=False)
    print(f"  ✓ Saved results to: {filename}")
    print(f"  ✓ Generated {len(summaries)} summaries")
    
    # Show example
    print(f"\n  Example Summary:")
    print(f"    Title: {titles[0][:80]}...")
    print(f"    Summary: {summaries[0][:80]}...")

print("\n" + "="*80)
print("ALL APPROACHES COMPLETED")
print("="*80)

# ============================================================================
# ANALYSIS FUNCTION
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS")
print("="*80)


def analyze_results():
    """
    Analyze results from all approaches.
    Calculates average, std dev, min, max for each metric.
    Creates comparison plots.
    """
    
    approach_names = ['frequency', 'tfidf', 'lexrank', 'textrank', 't5']
    metrics = ['fuzzy_score', 'jaccard_score', 'fasttext_score', 
               'rouge1_score', 'rouge2_score', 'rougeL_score']
    
    # Load all results
    all_results = {}
    for approach in approach_names:
        filename = f'results_{approach}.csv'
        try:
            df = pd.read_csv(filename)
            all_results[approach] = df
            print(f"Loaded {filename}")
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping...")
    
    if not all_results:
        print("No result files found!")
        return
    
    # Calculate statistics for each approach and metric
    statistics = []
    
    for approach in approach_names:
        if approach not in all_results:
            continue
            
        df = all_results[approach]
        
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].values
                
                stats = {
                    'Approach': approach.upper(),
                    'Metric': metric.replace('_score', '').upper(),
                    'Average': np.mean(values),
                    'Std Dev': np.std(values),
                    'Min': np.min(values),
                    'Max': np.max(values)
                }
                statistics.append(stats)
    
    # Create statistics DataFrame
    stats_df = pd.DataFrame(statistics)
    
    # Save statistics
    stats_df.to_csv('comprehensive_statistics.csv', index=False)
    print("\n✓ Saved statistics to: comprehensive_statistics.csv")
    
    # Print statistics table
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(stats_df.to_string(index=False))
    
    # Create visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Get data for this metric
        metric_data = stats_df[stats_df['Metric'] == metric.replace('_score', '').upper()]
        
        if len(metric_data) == 0:
            continue
        
        approaches_list = metric_data['Approach'].values
        averages = metric_data['Average'].values
        std_devs = metric_data['Std Dev'].values
        
        # Create bar plot with error bars
        x_pos = np.arange(len(approaches_list))
        bars = ax.bar(x_pos, averages, yerr=std_devs, 
                     capsize=5, alpha=0.7, color='steelblue')
        
        # Customize plot
        ax.set_xlabel('Approach', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_score", "").upper()} Scores', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(approaches_list, rotation=45, ha='right')
        ax.set_ylim([0, max(averages) * 1.2])
        
        # Add value labels on bars
        for bar, avg in zip(bars, averages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{avg:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plots to: comparison_plots.png")
    plt.close()
    
    # Create summary comparison plot (average across all metrics)
    print("\nCreating overall comparison plot...")
    
    overall_scores = []
    for approach in approach_names:
        if approach not in all_results:
            continue
        
        df = all_results[approach]
        
        # Calculate average across all metrics
        metric_values = []
        for metric in metrics:
            if metric in df.columns:
                metric_values.extend(df[metric].values)
        
        if metric_values:
            overall_scores.append({
                'Approach': approach.upper(),
                'Overall Average': np.mean(metric_values),
                'Overall Std': np.std(metric_values)
            })
    
    overall_df = pd.DataFrame(overall_scores)
    
    # Plot overall comparison
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(overall_df))
    bars = plt.bar(x_pos, overall_df['Overall Average'], 
                   yerr=overall_df['Overall Std'],
                   capsize=7, alpha=0.7, color='darkgreen')
    
    plt.xlabel('Approach', fontsize=14, fontweight='bold')
    plt.ylabel('Average Score (All Metrics)', fontsize=14, fontweight='bold')
    plt.title('Overall Performance Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x_pos, overall_df['Approach'], rotation=45, ha='right')
    plt.ylim([0, overall_df['Overall Average'].max() * 1.2])
    
    # Add value labels
    for bar, avg in zip(bars, overall_df['Overall Average']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('overall_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved overall comparison to: overall_comparison.png")
    plt.close()
    
    # Print best performing approach for each metric
    print("\n" + "="*80)
    print("BEST PERFORMING APPROACHES")
    print("="*80)
    
    for metric in metrics:
        metric_data = stats_df[stats_df['Metric'] == metric.replace('_score', '').upper()]
        if len(metric_data) > 0:
            best_idx = metric_data['Average'].idxmax()
            best_approach = metric_data.loc[best_idx]
            print(f"\n{metric.replace('_score', '').upper()}:")
            print(f"  Best: {best_approach['Approach']}")
            print(f"  Score: {best_approach['Average']:.4f} (±{best_approach['Std Dev']:.4f})")
    
    return stats_df, overall_df


# Run analysis
try:
    stats_df, overall_df = analyze_results()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. results_frequency.csv")
    print("  2. results_tfidf.csv")
    print("  3. results_lexrank.csv")
    print("  4. results_textrank.csv")
    print("  5. results_t5.csv")
    print("  6. comprehensive_statistics.csv")
    print("  7. comparison_plots.png")
    print("  8. overall_comparison.png")
    
except Exception as e:
    print(f"Error during analysis: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)