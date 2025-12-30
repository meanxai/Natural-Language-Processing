# [MXNLP-1-02] 2.nplm(emb).py
# A Neural Probabilistic Language Model (NPLM)
# Yoshua Bangio, et. al., 2003, A Neural Probabilistic Language Model
# Using the Embedding layer in Keras.
#
# This code is used in the Natural Language Processing (NLP)
# online course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/Natural-Language-Processing
#
# A detailed description of this code can be found in
# https://youtu.be/2nhhdz41ff8
#
import numpy as np
from tensorflow.keras.layers import Input, Dense, Add, Embedding
from tensorflow.keras.layers import Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.util import ngrams
import matplotlib.pyplot as plt

sentences = ["the cat is walking in the bedroom",
             "a dog was running in a room",
             "the cat is running in a room",
             "a dog is walking in a bedroom",
             "the dog was walking in the room"]

# Data preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word2idx = tokenizer.word_index
idx2word = {v: k for (k, v) in word2idx.items()}
sequences = tokenizer.texts_to_sequences(sentences) # word sequences

n = 4   # 4-grams
list_ngrams = [list(ngrams(s, n)) for s in sequences]
arry_ngrams = np.array(list_ngrams).reshape(-1, n)

# Training data
x_train = arry_ngrams[:, :-1]
y_train = arry_ngrams[:, -1].reshape(-1, 1)

m = 8                    # word embedding vector size
V = len(word2idx) + 1    # vocabulary size
h = 64                   # the number of hidden units

# Create a NPLM model
x_input = Input(batch_shape = (None, x_train.shape[1]), dtype='int32')
x_embed = Embedding(input_dim=V, output_dim=m, name='emb')(x_input)
x_concat = Flatten()(x_embed)
Hx = Dense(h, activation='tanh')(x_concat)  # tanh(d + Hx)
Wx = Dense(V, use_bias=False)(x_concat)     # Wx (optional)
Ux = Dense(V)(Hx)                           # b + U.tanh(d + Hx)
y = Add()([Wx, Ux])                         # Wx + b + U.tanh(d + Hx)
y_prob = Activation('softmax')(y)

model = Model(x_input, y_prob)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.01))
model.summary()

# Fit the data to the model
hist = model.fit(x_train, y_train, epochs=100)

# Loss history
plt.plot(hist.history['loss'], color='red')
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Matrix C
C = model.get_layer('emb').get_weights()[0]

# Word vectors
print("\nWord vectors:")
for word, idx in word2idx.items():
    word_vector = C[idx, :]
    print("{:10s}".format(word), end='')
    print(word_vector.round(3))

# P(w3 | w1, w2, w3)
print("\nP(w4 | w1, w2, w3):")
x = x_train[0]  # "the", "cat", "is"
p = model.predict(x.reshape(1, -1), verbose=0)[0]
print(idx2word[x[0]], idx2word[x[1]], idx2word[x[2]], ':')
for word, idx in word2idx.items():
    print("{:10s}: {:.3f}".format(word, p[idx]))

# A sentence matrix
print("\nSentence matrix:")
i = 0
print('Sentence:', sentences[i], ':')
print(C[sequences[i]].round(2))
