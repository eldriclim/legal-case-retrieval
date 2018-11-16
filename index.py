#!/usr/bin/python
import re
import sys
import getopt
import os
import bisect
import string
import csv
from time import time
from math import sqrt, log10
try:
    import cPickle as pickle
except:
    import pickle

from collections import Counter
from math import sqrt, log10


import nltk
from nltk import Text
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

start_time = time()

def set_csv_limit():
    maxlimit = sys.maxsize
    isLarge = True

    while isLarge:
        try:
            csv.field_size_limit(maxlimit)
            isLarge = False
        except OverflowError:
            maxlimit = int(maxlimit/2)

def index_data(dataset_filename, dictionary_filename, postings_filename):
    master_dict_postings = {}
    master_index_norm = {}
    index_docID_map = {}
    N = 0

    index_text_offset = {}
    index_nltk_text = {}

    # Set csv read limit
    set_csv_limit()

    with open(dataset_filename,'r') as infile:
        line = csv.reader(infile)
        row = next(line) # skip header


        for row in line:

            docID, title, content, date, court = row
            docID = int(docID)

            # Skip duplicate entries in dataset
            if docID in index_docID_map:
                continue

            index_docID_map[docID] = N

            # Format content and standardize format
            tokens = nltk.word_tokenize(content.translate(None, string.punctuation).decode("utf-8").encode("ascii","ignore"))

            # Apply case-folding & strip punctuations
            lemma_tokens = list(map(lemmatizer.lemmatize, \
                filter(lambda t: t.isalpha() and t not in stopwords, \
                    map(lambda token: token.lower(), tokens))))


            """
            To be used for context matching
            Stopwords were stripped to reduce space which may adversely affect context matching
            However, it will result in the decrease of false positives as adjacent words are
            now more significant. (holds some meaning in reference to query)
            """
            index_nltk_text[N] = " ".join(lemma_tokens)


            doc_lemma_count = {}
            doc_lemma_count_position = {}

            # Read content word by word
            for index, lemma in enumerate(lemma_tokens):
                if lemma not in doc_lemma_count_position:
                    doc_lemma_count_position[lemma] = [0,[]]
                    doc_lemma_count[lemma] = 0

                # Increment doc tf count; to be used for calculating normalizing factor
                doc_lemma_count[lemma] += 1
                # Increment doc tf count
                doc_lemma_count_position[lemma][0]+=1
                # Add positional index
                doc_lemma_count_position[lemma][1].append(index)

            # Calculate normalizing factor
            master_index_norm[N] = compute_doc_weight(doc_lemma_count.values())


            for lemma, post in doc_lemma_count_position.items():
                if lemma not in master_dict_postings:
                    master_dict_postings[lemma] = []

                # Insert posting while mataining sort integrity
                bisect.insort(master_dict_postings[lemma], ((docID, post[0], post[1])))

            # Increment count
            N += 1


        # Writing data into output file as string
        post_handle = open(postings_filename, 'w')
        with open(dictionary_filename, 'w') as dict_handle:
            for lemma in master_dict_postings:
                postings = master_dict_postings[lemma]
                postings = [(index_docID_map[docID], tf, positions) for docID, tf, positions in postings]

                master_dict_postings[lemma] = (len(postings), int(post_handle.tell()))
                #pickle.dump(posting, post_handle, protocol=pickle.HIGHEST_PROTOCOL)
                post_handle.write(generate_skip_list(postings))

            index_docID_map = dict((v,k) for k,v in index_docID_map.items())

            dict_handle.write(str(N)+"\n")

            dict_handle.write(" ".join([lem+":"+str(tuple[0])+":"+str(tuple[1]) for lem, tuple in master_dict_postings.items()])+"\n")
            dict_handle.write(" ".join([str(index)+":"+str(doc) for index, doc in index_docID_map.items()])+"\n")
            dict_handle.write(" ".join([str(index)+":"+str(norm) for index, norm in master_index_norm.items()])+"\n")


            """
            The following procedure indexes tokens for context matching
            May be commented off if doing either basic search or manual thesaurus query expansion
            """
            for index, text in index_nltk_text.items():
                index_text_offset[index] = int(post_handle.tell())
                post_handle.write(text+"\n")

            dict_handle.write(" ".join([str(index)+":"+str(offset) for index, offset in index_text_offset.items()]))
            """
            End of procedure
            """
        post_handle.close()


def generate_skip_list(postings):
	"""
	Takes a list of postings and converts it to a skip list string of postings with 'skip pointers'.
	Example:
	Input - ['10', '23', '74', '95', '100', '443', '736', '7014', '14736']
	Output - 1 +6 23 74 95 +8 100 443 736 7014 14736
	A skip pointer is indicated by the prefix "+", and its value indicates the number of
	bytes to add to the current offset to skip to the next posting.
	In this example, "+6" would indicate that the next posting is 6 bytes after the whitespace following skip pointer [95]
	"""

	# heuristic for skip factor
	skip_factor = int(sqrt(len(postings)))
	count = 0
	skip_list = ""

	for post in postings:
		skip_list += post_to_text(post) + " "
		count += 1
		if skip_factor == 1:
			continue
		else:
			if count % skip_factor == 1 and count + skip_factor <= len(postings):
				skip_bytes = get_skip_bytes(postings, count, skip_factor)
				skip_list += "+" + str(skip_bytes) + " "
	return skip_list.strip() + "\n"

def get_skip_bytes(postings, count, skip_factor):
	'''
	Calculates the number of bytes required for the skip pointer
	to point to the next posting from its current position
	'''
	temp = ""
	end = count + skip_factor - 1
	for i in range(count, end):
		temp += post_to_text(postings[i]) + " "

	return len(temp)

# Converts post to text representation
def post_to_text(post):
    return str(post[0])+":"+str(post[1])+":"+ \
        ",".join([str(p - post[2][i-1]) if i > 0 else str(p) for i, p in enumerate(post[2])])

def generate_files(data):
    (master_index, collection, dictionary, postings) = data
    master_dictionary = {}
    byte_offset = 0
    for term in master_index:
        postings_list = master_index[term]
        postings_str = get_postings_string(postings_list)
        df = len(postings_list)
        idf = log10(float(len(collection)) / float(df))
        # maintain a dictionary of terms with their offsets and idf to the posting list
        master_dictionary[term] = (byte_offset, idf)
        postings.write(postings_str)
        byte_offset += len(postings_str)

    # dump the dictionary object into dictionary file
    pickle.dump(master_dictionary, dictionary)
    pickle.dump(collection, dictionary)
    postings.close()

def get_postings_string(postings):

    """
    Takes a list of postings and convert it to the following format
    "doc_id1 normalized_log_weighted tf1 ... doc_idn normalized_log_weighted tfn"
    Example:
    Input - [(23, 0.065227275076), (74, 0.0547913372813)]
    Output - 23 0.065227275076 74 0.0547913372813
    """

    postings_string = ""
    for posting in postings:
        doc_id = posting[0]
        nlw_tf = str(posting[1])
        postings_string += doc_id + " " + nlw_tf + " "

    return postings_string.strip() + "\n"


def compute_doc_weight(counts):
    log_tf = [(1+log10(i))**2 for i in filter(lambda c: c > 0, counts)]
    return sqrt(reduce(lambda x,y: x+y, log_tf))

def usage():
    print ("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")

input_dataset = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except (getopt.GetoptError, err):
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_dataset = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_dataset == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

with open('stopwords.txt') as f:
    stopwords = set(map(lambda ln: ln.strip(), f.readlines()))

index_data(input_dataset, output_file_dictionary, output_file_postings)
