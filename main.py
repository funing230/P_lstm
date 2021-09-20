import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
from gensim.models import Word2Vec

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot

from nltk.stem.porter import PorterStemmer
from tensorflow import keras
from keras.preprocessing import sequence

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
from sklearn.preprocessing import LabelEncoder

import sys
sys.setrecursionlimit(10000)

#read bugreport for XLSX
projectname = 'Eclipse_Platform_UI_bugreport'
Eclipse_Platform_UI = pd.read_excel('dataset/' + projectname + '.xlsx', engine='openpyxl',nrows=5)
Eclipse_Platform_UI.rename(columns = {'Unnamed: 10' : 'result'}, inplace = True)

#read weight table
weight=pd.read_excel('dataset/' +'writer.xlsx', engine='openpyxl',header=None,nrows=5)
#processing weight
weight= weight.fillna(0)

p_weight=pd.get_dummies(weight,drop_first=True)

lastweight=preprocessing.scale(p_weight.iloc[:,1:])
lastweight= pd.DataFrame(lastweight);
lastweight.insert(0,'bug_id',1, allow_duplicates=False)
lastweight['bug_id']=weight[0]

lastweight= lastweight.drop('bug_id', 1)

print(lastweight.head(2))
#read Serverity for XLSX
serverity = pd.read_excel('dataset/' + 'serverity' + '.xlsx', engine='openpyxl',nrows=5)
serverity.rename(columns = {'Unnamed: 10' : 'result'}, inplace = True)

#concat serverity with bugreport
#Eclipse_Platform_UI=pd.concat([Eclipse_Platform_UI, lastweight,serverity], axis=1,keys='bug_id')
Eclipse_Platform_UI=pd.concat([Eclipse_Platform_UI, serverity], axis=1,keys='bug_id')
print(Eclipse_Platform_UI.head(10))


print(type(Eclipse_Platform_UI))
#get bugid summary description
df_id_br_s=Eclipse_Platform_UI.iloc[:,[1,2,3,-1]]


#labale data y
le = LabelEncoder()
y= le.fit_transform(df_id_br_s.iloc[:,-1])

#df_br_s.dropna(inplace = True)

#df.shape

## Get the Independent Features
X = df_id_br_s

#for word2vec dictionary
messages = (df_id_br_s.iloc[:,1]+df_id_br_s.iloc[:,2]).to_frame()

## Get the serverity  features
#y=df_id_br_s.iloc[:,-1]


### Vocabulary size
voc_size=5000


messages.reset_index(inplace = True)
#messages['title']

#Download the nltk toolkit
#nltk.download('stopwords')
#nltk.download('punkt')


### Dataset Preprocessing
def process_data(messages):
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages[0][i])
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


MAX_SEQUENCE_LENGTH = 500
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



# id summary description serverity

summary_description_word2vec_idex=tokenizer(df_id_br_s.iloc[:,1]+df_id_br_s.iloc[:,2], word_index)
print(summary_description_word2vec_idex.shape)
print(lastweight.shape)


main_input = keras.Input(shape=(None,), name="bugreport")
weight_input = keras.Input(shape=(lastweight.shape[1],), name="weight")

main_features =Embedding(output_dim=embeddings_matrix.shape[0], input_dim=embeddings_matrix.shape[1], input_length=300)(main_input)

lstm_out = LSTM(64)(main_features)

x = keras.layers.concatenate([lstm_out, weight_input])
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
#x = tf.keras.layers.Flatten(x)

output = Dense(1, activation='softmax', name='output')(x)

model = Model(inputs=[main_input, weight_input], outputs=output)
model.compile(optimizer='rmsprop',
loss={'output': keras.losses.CategoricalCrossentropy(from_logits=True)},
loss_weights={'output': 0.2})

print(model.summary())

main_input = summary_description_word2vec_idex
aux_input = lastweight

# 通过字典的形式将数据fit到模型
model.fit(
    {"bugreport": summary_description_word2vec_idex, "weight": lastweight},
    {"output": y},
    epochs=10,
    batch_size=32,
)

#keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

# #main_input = summary_description_word2vec_idex
#
# main_input = Input((500,), dtype='float64', name='main_input')
#
# x = Embedding(output_dim=embeddings_matrix.shape[0], input_dim=embeddings_matrix.shape[1], input_length=300)(main_input)
#
# lstm_out = LSTM(64)(x)
#
# aux_input = Input((lastweight.shape[1],), name='aux_input')
#
# x = keras.layers.concatenate([lstm_out, aux_input])
# x = Dense(64, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# #x = tf.keras.layers.Flatten(x)
# #model.add(tf.keras.layers.Flatten())
# main_output = Dense(1, activation='softmax', name='main_output')(x)
#
# model = Model(inputs=[main_input, aux_input], outputs=main_output)
# model.compile(optimizer='rmsprop',
# loss={'main_output': keras.losses.CategoricalCrossentropy(from_logits=True)},
# loss_weights={'main_output': 0.2})
# print(model.summary())

main_input = summary_description_word2vec_idex
aux_input = lastweight


# # Dummy target data
# priority_targets = np.random.random(size=(1280, 1))
# # dept_targets = np.random.randint(2, size=(1280, num_departments))


#







#
# #X = tokenizer(sentences, word_index) #texts is numpy, input into the model calculation.
#
# #embeddings_matrix.shape
#
# model = keras.Sequential([
#       keras.layers.Embedding(input_dim=embeddings_matrix.shape[0],
#                              output_dim=embeddings_matrix.shape[1],
#                              weights=[embeddings_matrix],
#                              input_length=20),
#       keras.layers.LSTM(200),
#       keras.layers.Dropout(0.3),
#       keras.layers.Dense(1, activation='sigmoid')
#    ])
# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#
# model.summary()
#
#
# X_final=X   #np.array(X)
# y_final=np.array(y)
#
# X_final.shape,y_final.shape
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.30, random_state=42)
#
# ### Finally Training
# model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64)
#
#
# y_pred=(model.predict(X_test)>=0.5).astype("int")   # use sigmoid value
#
#
# from sklearn.metrics import confusion_matrix
#
# confusion_matrix(y_test,y_pred)
#
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test,y_pred))
#
#
#
#

