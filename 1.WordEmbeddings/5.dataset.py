# [MXNLP-1-04] 5.dataset.py
# Preprocess the Gutenberg Corpus and store the results 
# for later use.
#
# This code is used in the Natural Language Processing (NLP)
# online course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/Natural-Language-Processing
#
# A detailed description of this code can be found in
# https://youtu.be/_P55dYohlA4
#
import numpy as np
import nltk

from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer

# Download the Gutenberg corpus using NLTK.
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('stopwords')

# Define the stop words
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(("''", "``", "--", "'s", "n't"))

# List of 18 files in the Gutenberg Corpus.
files = nltk.corpus.gutenberg.fileids()

# Count word frequencies within the 18 text files
word_freq = Counter()
list_sentences1 = []   # A list for storing a collection of sentences 
                       # excluding stop words.
for i, file in enumerate(files):
    text = nltk.corpus.gutenberg.raw(file)
    sentences = nltk.sent_tokenize(text.lower())

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [w for w in words if w not in stopwords if len(w) > 1]
        if len(words) >= 5:
            word_freq += Counter(words)
            list_sentences1.append(words)
    print('{}: {} -----> done.'.format(i+1, file))
sum_freq = word_freq.total()  # the sum of word frequencies

# the probability of keeping the word
def sampling_rate(word):
    f = word_freq[word] / sum_freq
    return (np.sqrt(f / 1e-3) + 1) * 1e-3 / f

# Subsampling of frequent words
list_sentences2 = []   # A list for storing a collection of sentences 
                       # with subsampling applied.
for sentence in list_sentences1:
    words = [word for word in sentence \
             if np.random.rand() < sampling_rate(word)]
    if len(words) >= 5:
        list_sentences2.append(words)

# Tokenize the collection of the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list_sentences2)

# Create a vocabulary
word2idx = tokenizer.word_index
word2idx['<PAD>'] = 0
idx2word = {v:k for k, v in word2idx.items()}

# Convert the words in the sentences into their corresponding 
# indices in the vocabulary.
sequences = tokenizer.texts_to_sequences(list_sentences2)

# Store the vocabulary and the sequences for later use
import pickle
with open('data/gutenberg.pkl', 'wb') as f:
 	pickle.dump([word2idx, idx2word, sequences], f)
