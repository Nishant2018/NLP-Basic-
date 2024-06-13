# Basics of Natural Language Processing (NLP)

## Introduction

Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It involves the application of computational techniques to analyze and synthesize natural language and speech. NLP has a wide range of applications including sentiment analysis, machine translation, chatbots, and more.

## Key Concepts

### Text Preprocessing

Text preprocessing is the first step in NLP. It involves cleaning and preparing text data for analysis. Common preprocessing steps include:

- **Tokenization**: Splitting text into individual words or tokens.
- **Lowercasing**: Converting all text to lowercase to maintain consistency.
- **Removing Punctuation and Stopwords**: Filtering out non-essential words and punctuation.
- **Lemmatization/Stemming**: Reducing words to their base or root form.

### Tokenization

Tokenization is the process of breaking down text into smaller units called tokens (words, phrases, symbols).

```python
from nltk.tokenize import word_tokenize

text = "Natural Language Processing is fascinating!"
tokens = word_tokenize(text)
print(tokens)
```
### Stopwords Removal

```python
from nltk.corpus import stopwords

tokens = ["Natural", "Language", "Processing", "is", "fascinating"]
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(filtered_tokens)
```

### Lemmatization and Stemming

```python
from nltk.stem import WordNetLemmatizer, PorterStemmer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

word = "running"
lemma = lemmatizer.lemmatize(word, pos='v')
stem = stemmer.stem(word)

print(f"Lemma: {lemma}, Stem: {stem}")
```

### Feature Extraction

#### Bag of Words (BoW)

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "Natural Language Processing is fascinating.",
    "I love studying NLP.",
    "Language models are essential in NLP."
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())
```

#### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())
```
