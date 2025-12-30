# [MXNLP-1-04] 6.CBOW.py
# Implementation of a CBOW model
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
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle

# Load the preprocessed Gutenberg Corpus
with open('data/gutenberg.pkl', 'rb') as f:
    word2idx, idx2word, sequences = pickle.load(f)

# Create a training dataset with 5-grams
n_grams = 5
x_train = []
y_train = []
for sequence in sequences:
    if len(sequence) >= n_grams:
        for a, b, c, d, e in nltk.ngrams(sequence, n_grams):
            x_train.append([a, b, d, e])
            y_train.append(c)

x_train = np.array(x_train)
y_train = np.array(y_train).reshape(-1, 1)

n_vocab = len(word2idx)   # vocabulary size
n_emb = 32                # word embedding vector size

# Custom Embedding layer
class WordEmbedding(Layer):
    def __init__(self, input_dim, emb_size, name=None):
        super(WordEmbedding, self).__init__(name = name)
        
        # Matrix C
        C_init = tf.random_normal_initializer()
        self.C = tf.Variable(
            initial_value = C_init(shape=(input_dim, emb_size)),
            trainable = True,
        )
        
    def call(self, x):
        # Table look-up in C
        selected_rows = tf.gather(self.C, x)  # (None, 4, 32)
        return selected_rows
    
# Create a CBOW model
x_input = Input(batch_shape=(None, x_train.shape[1]), dtype='int32')
x_emb = WordEmbedding(n_vocab, n_emb, name='emb')(x_input)
x_avg = tf.reduce_mean(x_emb, axis=1)
y_output = Dense(n_vocab, activation='softmax')(x_avg)

model = Model(x_input, y_output)
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer='adam')
model.summary()

# Fit the model to the training dataset
hist = model.fit(x_train, y_train,
                 batch_size=20480,
                 shuffle = True,
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

