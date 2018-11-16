#!/usr/bin/python2
import re
import sys
import getopt
import nltk
from nltk import stem
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from skiplist import SkipList
import string
import heapq
from math import sqrt, log10
from collections import Counter


try:
   import cPickle as pickle
except:
   import pickle

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

def search_and_rank():

    # Queries for search, assumed boolean presence instead of count
    # bool_query: words/phrase in Boolean Retrieval
    # free_query: words in Free Text Search
    # total_query: all words in query
    bool_query, free_query, total_query = read_query_line(file_of_queries)


    # Result from Boolean Retrieval
    # To be ranked higher than the rest
    filtered_docID = run_boolean_retrieval(bool_query)


    # Initialising term-idf mapping
    lemma_idf = {}
    for query in total_query:
        if query not in master_dict:
            continue
        query_df = master_dict[query][0]
        query_idf = log10(float(N)/float(query_df))
        lemma_idf[query] = query_idf


    # Retrieve WordNet synonyms for query expansion
    # Disjoint set of words from 'total_query', to be used for expansion
    lemma_syn_set = retrieve_wordnet_synonyms(total_query)


    """
    Pseudo Relevance Feedback
    """
    """
    # Significant documents for PRF
    sig_doc = set(filtered_docID)

    # If Free Text Search: calculate preliminary Cosine Similarity for relevant docs
    # If Boolean Retrieval: skip this step, unless size of filtered doc is too large (>50),
    # this is to minimise number of relevant doc vectors for faster runtime
    if len(sig_doc) > 50 or len(sig_doc) == 0:
        docID_score = calculate_cosine_score(total_query, lemma_idf)
        for docID, score in docID_score.items():
            if score > 0:
                docID_score[docID] = float(score)/master_docID_norm[master_docID_index_map[docID]]

        # Extract top 10 relevant doc based on preliminary Cosine Similarity
        sig_doc = heapq.nlargest(10, docID_score, key=docID_score.get)

    # Retrieve summation of all relevant doc vectors
    relevant_vector_sum = retrieve_relevant_doc_vector(sig_doc)

    # Apply idf, normalise and Rocchio's weight
    rocchio_relevant = {k:((float(v)/len(sig_doc))*(log10(float(N)/float(master_dict[k][0])))*0.5) for k,v in relevant_vector_sum.items()}
    identified = heapq.nlargest(5, rocchio_relevant, key=rocchio_relevant.get)

    doc_sub_vector = {i:rocchio_relevant[i] for i in identified}

    # Update term-idf mapping for Cosine Similarity later (Q+D vector addtion in tf-idf form)
    # No longer in boolean presence tf-idf form
    for k,v in rocchio_relevant.items():
        if k in lemma_idf:
            lemma_idf[k] += v
        else:
            lemma_idf[k] = v
    """
    """
    End of Pseudo Relevance Feedback
    """

    """
    Manual thesaurus using WordNet
    """
    """
    # Add all synonyms with similar definition for expansion
    total_query = total_query.union(lemma_syn_set)
    """
    """
    End of Manual thesaurus
    """


    """
    Auto thesaurus (w/ definition-filtering) using NLTK's ContextIndex
    Documents from Boolean Retrieval are used as base text as it allows
    for a more accurate representation of user search space
    """
    """
    sig_doc = set(filtered_docID)

    # Using Boolean Retrieved docs as our base text for context search,
    # we narrow our search scope for improved accuracy and runtime
    if len(sig_doc) > 50 or len(sig_doc) == 0:
        docID_score = calculate_cosine_score(total_query, lemma_idf)
        for docID, score in docID_score.items():
            if score > 0:
                docID_score[docID] = float(score)/master_docID_norm[master_docID_index_map[docID]]

        # Identify significant document
        sig_doc = heapq.nlargest(50, docID_score, key=docID_score.get)

        # Retrieve word with similar context from input queries
    lemma_sim_context = retrieve_similar_context_synonyms(sig_doc, total_query)

    # Extract only synonyms with similar definition and context for expansion
    lemma_syn_set = lemma_syn_set.intersection(set(lemma_sim_context))
    total_query = total_query.union(lemma_syn_set)
    """
    """
    End of Auto thesaurus
    """


    # Calculate cosine scores
    docID_score = calculate_cosine_score(total_query, lemma_idf)


    for docID, score in docID_score.items():
        if score > 0:
            docID_score[docID] = float(score)/master_docID_norm[master_docID_index_map[docID]]

    # Boolean Retrieved results should be of a higher-tier
    # Considering a cosine graph's maxima being 1,
    # pre-scoring the selected docID as 1 ensures
    # that they are ranked above non-selected docID
    for docID in filtered_docID:
        docID_score[docID] += 1

    filtered_dict = dict((v, k) for (k, v) in docID_score.iteritems() if v > 0)

    # The following ranks the docID by its scores and extracts the docID
    heap = [(-key, value) for key,value in filtered_dict.items()]
    heapq.heapify(heap)
    output = [(o[1]) for o in sorted(heap)]

    with open(file_of_output, 'w') as output_handle:
        output_handle.write(" ".join(map(str,output)))


def calculate_cosine_score(total_query, lemma_idf):
    # Initialise scores
    intermediate_docID_score = {master_index_docID_map[k]:0 for k in range(N)}

    # Term by term computation
    for query in total_query:
        if query not in master_dict:
            continue

        # Retrieve skip list for query
        lemma_list = get_posting(query)
        while lemma_list.peek() is not 'EOL':
            if not lemma_list.skippable():
                docID = master_index_docID_map[lemma_list.index()]
                doc_tf_weight = 1 + log10(float(lemma_list.tf()))

                # Evaluation step
                intermediate_docID_score[docID] += lemma_idf[query] * doc_tf_weight

            lemma_list.next()

    return intermediate_docID_score

# Assumes that weight of query is defaulted to 1
def read_query_line(file_of_queries):

    bool_query = []
    free_query = []

    with open(file_of_queries, 'r') as infile:
        total_lemma = []

        line = infile.readline().rstrip()
        line = line.translate(None, string.punctuation.replace("\"","")).rstrip()

        isBoolean = " AND " in line

        if isBoolean:
            # boolean query
            store = line.split(" AND ")
            store = list(map(str.lower, store))

            for element in store:
                if element.startswith("\""):
                    # case: phrase
                    phrase = element.replace("\"","")
                    phrase = " ".join(map(lemmatizer.lemmatize, phrase.split(" ")))
                    bool_query.append(phrase)
                    total_lemma += phrase.split(" ")
                else:
                    # case: word
                    lemma = lemmatizer.lemmatize(element)
                    bool_query.append(lemma)
                    total_lemma.append(lemma)
        else:
            line = line.lower()
            if line.startswith("\""):
                # boolean query: single phrase
                line = line.replace("\"","")
                phrase = " ".join(map(lemmatizer.lemmatize, line.split(" ")))
                bool_query.append(phrase)
                total_lemma += phrase.split(" ")
            else:
                # free-text query
                free_query = set(map(lemmatizer.lemmatize, line.split(" ")))
                total_lemma = free_query

        bool_query = set(bool_query)
        free_query = set(free_query)
        total_lemma = set(total_lemma)


        relevant_doc = []


        for line in infile:
            relevant_doc.append(line.rstrip())


    return (bool_query, free_query, total_lemma)

def retrieve_wordnet_synonyms(total_lemma):
    lemma_syn = []

    # Retrieve words with similar definition
    for lemma in total_lemma:
        lemma_syn += sum(map(lambda syn: syn.lemma_names(), wordnet.synsets(lemma)), [])
    lemma_syn = map(lambda syn: syn.decode("utf-8").encode("ascii","ignore").lower(), filter(lambda lem: lem.isalpha(), lemma_syn))

    # Returns discovered words not in the initial search
    return set(lemma_syn) - set(total_lemma)

def retrieve_similar_context_synonyms(relevant_docID, total_query):

    synonyms = []
    text = ""

    # Load text from relevant docs as base text for context matching
    for docID in relevant_docID:
        index = master_docID_index_map[docID]

        with open(dictionary_file, 'r') as dict_handle, open(postings_file, 'r') as post_handle:
            post_handle.seek(master_index_text_offset[index])
            text += post_handle.readline().rstrip()

    nltk_text = nltk.ContextIndex(text)

    # Obtain list of similar contexted words from query
    for query in total_query:
        synonyms += nltk_text.similar_words(query)

    synonyms = set(synonyms)
    return synonyms

# Does boolean search, similar to HW2
def run_boolean_retrieval(bool_query):
    results = []

    for search in bool_query:
        if " " in search:
            # phrase detected
            slists = list(map(lambda word: get_posting(word) ,
                        filter(lambda s: s in master_dict , search.split(" "))))

            if len(search.split(" ")) is not len(slists):
                # phrase words not found
                return []
            sub_result = get_phrasal_search(slists)
            sub_result = sorted(map(int, sub_result))

            if not results:
                # first computation
                if sub_result:
                    results = sub_result
                else:
                    return []
            else:
                # phrasal search when results exist

                if sub_result:
                    results = set(results).intersection(set(sub_result))
                    if not results:
                        # terminates when intersection evaluates to empty
                        return []
                else:
                    return []

        else:
            # word detected
            if search not in master_dict:
                # terminates when word not in dictionary
                return []

            slist = get_posting(search)

            if not results:
                # first computation
                while slist.peek() is not 'EOL':
                    if not slist.skippable():
                        post = slist.peek()
                        results.append(master_index_docID_map[post[1][0]])
                    slist.next()

            else:
                results = sorted(results)

                new_results = []
                max = len(results)
                curr = 0

                slist1 = get_posting(search)
                r = []
                while slist1.peek() is not 'EOL':
                    if not slist1.skippable():
                        r.append(master_index_docID_map[slist1.index()])
                    slist1.next()

                while curr != max and slist.peek() is not 'EOL':
                    s = master_index_docID_map[slist.index()]
                    if slist.skippable():
                        if s < results[curr]:
                            slist.skip(slist.skip_offset())
                            continue
                        else:
                            slist.next()
                            continue
                    if s < results[curr]:
                        slist.next()
                    elif s > results[curr]:
                        curr+=1
                    else:
                        new_results.append(s)
                        curr+=1
                        slist.next()
                if new_results:
                    results = new_results
                else:
                    return []
    return results

# Does boolean search, similar to HW2
# Works only for 2 and 3 word phrases
def get_phrasal_search(slists):
    listings = []

    if len(slists) == 2:
        first = slists[0]
        second = slists[1]
        while first.peek() is not 'EOL' and second.peek() is not 'EOL':
            f = master_index_docID_map[first.index()]
            s = master_index_docID_map[second.index()]
            if first.skippable():
                if f < s:
                    first.skip(first.peek()[2])
                else:
                    first.next()
                continue
            if second.skippable():
                if f > s:
                    second.skip(second.peek()[2])
                else:
                    second.next()
                continue
            if f < s:
                first.next()
            elif f > s:
                second.next()
            else:
                # Adds offset and check for similarity
                a = set(map(lambda i: i+1, first.position()))
                b = set(second.position())

                if len(a.intersection(b)) > 0:
                    listings.append(f)
                first.next()
                second.next()


    if len(slists) == 3:
        first = slists[0]
        second = slists[1]
        third = slists[2]

        while first.peek() is not 'EOL' and second.peek() is not 'EOL' and third.peek() is not 'EOL':
            if first.skippable():
                first.next()
            if second.skippable():
                second.next()
            if third.skippable():
                third.next()

            f = master_index_docID_map[first.index()]
            s = master_index_docID_map[second.index()]
            t = master_index_docID_map[third.index()]

            if f < s or f < t:
                first.next()
                continue
            if s < f or s < t:
                second.next()
                continue
            if t < f or t < s:
                third.next()
                continue

            if f == s == t:
                # Adds offset and check for similarity
                a = set(map(lambda i: i+2, first.position()))
                b = set(map(lambda i: i+1, second.position()))
                c = set(third.position())

                if len(a.intersection(b).intersection(c)) > 0:
                    listings.append(f)

                first.next()
                second.next()
                third.next()

    return listings

# Retrieves posting list from term
def get_posting(lemma):
    if lemma in master_dict:
        return SkipList(open(postings_file), master_dict[lemma][1])
    else:
        return False

# For PRF, retrieval of tf for relevant docs
def retrieve_relevant_doc_vector(relevant_docID):

    # Master Count maintains the tf of all relevant document
    master_count = Counter()

    # Count terms in all relevant docs
    for docID in relevant_docID:
        index = master_docID_index_map[docID]

        # Read doc contents
        with open(dictionary_file, 'r') as dict_handle, open(postings_file, 'r') as post_handle:
            post_handle.seek(master_index_text_offset[index])
            text = post_handle.readline().rstrip().replace("\n","")

        tokens = text.split(" ")

        # Count terms and collate to Master Count
        doc_vector = Counter(tokens)
        master_count += doc_vector

    return master_count


def usage():
    print ("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")



dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except (getopt.GetoptError, err):
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)


master_dict = {}
master_index_docID_map = {}

# Reads in input documents and loads necessary data structures
with open(dictionary_file, 'r') as dict_handle:
    N = int(dict_handle.readline().rstrip())

    string_dict = dict_handle.readline().rstrip().split(" ")
    string_dict = map(lambda entry: entry.split(":"), string_dict)
    master_dict = {data[0]:(int(data[1]),int(data[2])) for data in string_dict}

    string_index_map = dict_handle.readline().rstrip().split(" ")
    string_index_map = map(lambda entry: entry.split(":"), string_index_map)
    master_index_docID_map = {int(data[0]):int(data[1]) for data in string_index_map}

    string_norm = dict_handle.readline().rstrip().split(" ")
    string_norm = map(lambda entry: entry.split(":"), string_norm)
    master_docID_norm = {int(data[0]):float(data[1]) for data in string_norm}

    """
    Only parsed when running PFB or Auto thesaurus
    """
    string_text_offset = dict_handle.readline().rstrip().split(" ")
    string_text_offset = map(lambda entry: entry.split(":"), string_text_offset)
    master_index_text_offset = {int(data[0]):int(data[1]) for data in string_text_offset}
    """
    End of parsing
    """

master_docID_index_map = {int(v):int(k) for k,v in master_index_docID_map.items()}


search_and_rank()
