# Legal Case Retrieval

## Requirements
Python 2.7 is required.

## Objectives
Experiment different query expansion techniques for retrieving relevant legal documents.

## Indexing
1. Stop words and punctuations are stripped to remove noise
2. Words tokenized by `WordNetLemmatizer` instead of `porter` from `NLTK` library for better `SynSet` matching
3. Perform case folding
4. Apply compression techniques

Positional index is also recoded for phrasal searches.

## Searching
1. Setup underlying skip list for faster searches
2. Performs boolean retrieval and free text search
   * boolean retrieval searches to have higher precedence
3. Performs query expansion
4. Calculate cosine similarity score

### Query Expansion

* Pseudo Relevance Feedback
* Manual Thesaurus with WordNet
* Auto Thesaurus with ContextIndex
