# Text-Classification
# TEXT CLASSIFICATION
import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

EMBEDDING_FILE='rnn/glove.6B/glove.6B.100d.txt'
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

embed_size = 100 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 463

train.categories.unique()
train_labels = train['categories']
#test_labels = test['categories']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()                  # converts the character array to numeric array. Assigns levels to unique labels.
le.fit(train_labels)
train_labels = le.transform(train_labels)
#test_labels = le.transform(test_labels)

print(le.classes_)
print(np.unique(train_labels, return_counts=True))
train_labels=pd.DataFrame(train_labels)
train_labels.head() 
list_sentences_train = train["converse"].fillna("_na_").values

y = train_labels
list_sentences_test = test["converse"].fillna("_na_").values
pd.DataFrame(list_sentences_train).isnull().any()
pd.DataFrame(list_sentences_test).isnull().any()
from keras.utils import to_categorical
y=to_categorical(y)
y.shape
(~y.any(axis=0)).any()
y.shape
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
GLOVE_DIR = 'glove.6B/'

print('Indexing word vectors.')
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std
print(emb_mean)
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

from keras.models import Sequential
embedding_size = 100
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(LSTM(100))
model.add(Dense(21, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
model.fit(X_t, y, validation_split=0.2, epochs=3, batch_size=64)
test_pred = model.predict_classes(X_te)
test_pred
test_pred_classes = le.inverse_transform(test_pred)
df_test_pred_classes = pd.DataFrame(test_pred_classes)
df_test_pred_classes.to_csv("Predictions_embedded.csv")











