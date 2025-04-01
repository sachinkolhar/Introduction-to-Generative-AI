import transformers
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Install dependencies (Run this in a separate cell if needed)
!pip install transformers sentence-transformers

# Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
texts = [
    'I love this product! It was great.',
    'I am not happy with this service.',
    'This is the best book I have ever read.',
    'I am disappointed with the quality of the item'
]

results = sentiment_pipeline(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}\nSentiment: {result['label']}, Confidence: {result['score']:.4f}\n")

# Translation Pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
translated_texts = translator(texts)
for original, translated in zip(texts, translated_texts):
    print(f"Original: {original}\nTranslated: {translated['translation_text']}\n")

# Summarization Pipeline
summarizer = pipeline("summarization")
summarization_texts = [
    """
    The quick brown fox jumps over the lazy dog. This well-known
    English-language pangram—a sentence that contains all of the letters
    of the English alphabet—has been used to test typewriters and display fonts.
    Pangrams have been used in programming and testing since the earliest days of computing.
    """,
    """
    Artificial Intelligence, or AI, is a field of computer science that focuses on creating machines
    capable of performing tasks that typically require human intelligence.
    The field is constantly evolving, and researchers are continuously finding new ways to improve AI algorithms.
    """
]
summaries = summarizer(summarization_texts, max_length=50, min_length=25, do_sample=False)
for i, summary in enumerate(summaries):
    print(f"Summary {i + 1}: {summary['summary_text']}\n")

# Sentence Embeddings
model_name = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
embedding_model = SentenceTransformer(model_name)
embeddings = embedding_model.encode(texts)
for sentence, embedding in zip(texts, embeddings):
    print("Sentence:", sentence)
    print("Embedding Size:", len(embedding))
    print("Embedding:", embedding[:5], "... (truncated)")
    print()
