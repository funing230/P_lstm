import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from nltk.stem.porter import PorterStemmer
from tensorflow import keras
from keras.preprocessing import sequence


df=pd.read_csv('train.csv',nrows=1500)#,nrows=15000

df.dropna(inplace = True)

#df.shape

## Get the Independent Features
X = df.drop('label', axis = 1)

## Get the Dependent features
y=df['label']


### Vocabulary size
voc_size=5000

messages = X.copy()
messages.reset_index(inplace = True)
#messages['title']

#Download the nltk toolkit
nltk.download('stopwords')
nltk.download('punkt')


### Dataset Preprocessing
def process_data(messages):
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

#split word
sentences= [nltk.word_tokenize(words) for words in process_data(messages)]

#word2vec
EMBEDDING_LEN=100
def get_word2vec_dictionaries(texts):

    Word2VecModel =Word2Vec(texts, window=7, min_count=5, workers=4) #  Get the word2vector model
    words=list(Word2VecModel.wv.index_to_key)
    vocab_list = [word for word in words]  # Store all words  index_to_key enumerate(Word2VecModel.wv.index_to_key)


    word_index = {" ": 0}      # Initialize `[word: token]`, and later tokenize the corpus to use this dictionary.
    word_vector = {}           # Initialize the `[word: vector]` dictionary

    # Initialize , pay attention to one more bit (first row), the word vector is all 0, which is used for padding.
    # embeddings_matrix :The number of rows is the number of all words +1,
    # the number of columns is the "dimension" of the word vector, such as 100.
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

    ## Fill in the above dictionary and matrix
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # Every word
        word_index[word] = i + 1  #Words: serial number
        word_vector[word] = Word2VecModel.wv[word] #Words: word vectors
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # Word vector matrix

    return word_index, word_vector, embeddings_matrix


word_index, word_vector, embeddings_matrix = get_word2vec_dictionaries(sentences)



MAX_SEQUENCE_LENGTH = 20
# Serialize the text, tokenizer sentence, and return the word index corresponding to each sentence
def tokenizer(sentences, word_index):
    index_data = []
    for sentence in sentences:
        index_word = []
        for word in sentence:
            try:
                index_word.append(word_index[word])  # Convert the words to index
            except:
                index_word.append(0)
        index_data.append(index_word)

    #Use padding of kears to align sentences. The advantage is that the numpy array is output
    index_texts = sequence.pad_sequences(index_data, maxlen=MAX_SEQUENCE_LENGTH)
    return index_texts


X = tokenizer(sentences, word_index) #texts is numpy, input into the model calculation.

#embeddings_matrix.shape

model = keras.Sequential([
      keras.layers.Embedding(input_dim=embeddings_matrix.shape[0],
                             output_dim=embeddings_matrix.shape[1],
                             weights=[embeddings_matrix],
                             input_length=20),
      keras.layers.LSTM(200),
      keras.layers.Dropout(0.3),
      keras.layers.Dense(1, activation='sigmoid')
   ])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()


X_final=X   #np.array(X)
y_final=np.array(y)

X_final.shape,y_final.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.30, random_state=42)

### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64)


y_pred=(model.predict(X_test)>=0.5).astype("int")   # use sigmoid value


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))





