# [MXNLP-1-05] 8.SkipGram(2).py
# Implementation of a Skip-Gram model using a single output layer
#
# This code is used in the Natural Language Processing (NLP)
# online course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/Natural-Language-Processing
#
# A detailed description of this code can be found in
# https://youtu.be/Sod7E0h4I7Q
#
import numpy as np
import nltk
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle

# Load the preprocessed Gutenberg Corpus
with open('data/gutenberg.pkl', 'rb') as f:
    word2idx, idx2word, sequences = pickle.load(f)

# Create a training dataset
n_grams = 3
x_train = []
y_train = []
for sequence in sequences:
    if len(sequence) >= n_grams:
        for a, b, c in nltk.ngrams(sequence, n_grams):
            x_train.append(b)
            y_train.append(a)
            
            x_train.append(b)
            y_train.append(c)
            
x_train = np.array(x_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)

n_vocab = len(word2idx)   # vocabulary size
n_emb = 16                # word embedding vector size

# Creating a simple Skip-Gram model
x_input = Input(batch_shape=(None, 1))
x_emb = Embedding(n_vocab, n_emb, name='emb')(x_input)  # (None, 1, 32)
x_emb = Reshape((n_emb,))(x_emb)                        # (None, 32)
y_output = Dense(n_vocab, activation='softmax')(x_emb)  # (None, 51420)

model = Model(x_input, y_output)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam')
model.summary()

# Fit the model to the training dataset
hist = model.fit(x_train, y_train,
                 batch_size=20000,
                 epochs=300)

# Loss history
plt.plot(hist.history['loss'], color='red')
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Word embedding matrix C
C = model.get_layer('emb').get_weights()[0]

# To represent word embedding vectors in a two-dimensional 
# vector space, we use pca to transform the 32-dimensional matrix C 
# into a two-dimensional matrix.
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_C = pca.fit_transform(C)

# we represent the two-dimensional embedding vectors of 
# the following words in a two-dimensional space.
words = ['daffodil', 'marigold', 'iris', 'rose', 'violet',
         'france', 'netherlands', 'spain', 'turkey', 'ireland',
         'mother', 'father', 'son', 'daughter', 'family']
words_idx = np.array([word2idx[x] for x in words])
words_vec = pca_C[words_idx]

# Let's see if words with similar meanings are close together 
# in the space.
x = words_vec[:, 0]
y = words_vec[:, 1]
plt.figure(figsize=(8,6))
plt.scatter(x, y, s=200, c='red', alpha=0.7)
for i in range(len(words)):
    plt.annotate(words[i], (x[i], y[i] + 0.1))
# plt.xlim(-2.0, 1.4)    
# plt.ylim(-1.7, 1.4)
plt.show()