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





#read bugreport for XLSX
projectname = 'Eclipse_Platform_UI_bugreport'
Eclipse_Platform_UI = pd.read_excel('dataset/' + projectname + '.xlsx', engine='openpyxl',nrows=30)
Eclipse_Platform_UI.rename(columns = {'Unnamed: 10' : 'result'}, inplace = True)

#read weight table
weight=pd.read_excel('dataset/' + 'weight_table.xlsx', engine='openpyxl',header=None,nrows=30)
w_bug_id=weight.iloc[:,0]
weight= weight.iloc[:,1:].fillna(0)
maxmin = preprocessing.MinMaxScaler()
# print(weight.dtypes)

def is_simple_numpy_number(dtype):
    if np.issubdtype(dtype, np.integer):
        return True
    if np.issubdtype(dtype, np.floating):
        return True
    return False


for col in range(weight.shape[1]):
    if is_simple_numpy_number(weight.iloc[:, col].dtype):
        scale_param = maxmin.fit(weight.iloc[:, col].values.reshape(-1, 1))
        weight.iloc[:, col]=maxmin.fit_transform(weight.iloc[:, col].values.reshape(-1, 1),scale_param)

lastweight=pd.get_dummies(weight)

# lastweight.insert(0,'bug_id',1, allow_duplicates=False)
# lastweight['bug_id']=w_bug_id

print(lastweight.info())
print(lastweight.head(2))

#read Serverity for XLSX
serverity = pd.read_excel('dataset/' + 'serverity' + '.xlsx', engine='openpyxl',nrows=30)
serverity.rename(columns = {'Unnamed: 10' : 'result'}, inplace = True)

#concat serverity with bugreport

Eclipse_Platform_UI=pd.concat([Eclipse_Platform_UI, serverity], axis=1,keys='bug_id')
print(Eclipse_Platform_UI.head(10))


print(type(Eclipse_Platform_UI))
#get bugid summary description
df_id_br_s=Eclipse_Platform_UI.iloc[:,[1,2,3,-1]]


#labale data y

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
oe.fit(df_id_br_s.iloc[:,-1].values.reshape(-1, 1)).categories_

y_labels = oe.fit_transform(df_id_br_s.iloc[:,-1].values.reshape(-1, 1))


## Get the Independent Features


X=df_id_br_s.fillna(' ')
#for word2vec dictionary
#messages = (df_id_br_s.iloc[:,1]+df_id_br_s.iloc[:,2]).to_frame()
#messages = df_id_br_s.iloc[:,1] + np.where(df_id_br_s.iloc[:,2]!='no_item', ', '+df_id_br_s.iloc[:,2],'')
messages = (X.iloc[:,1].map(str)+X.iloc[:,2]).to_frame()
## Get the serverity  features
#y=df_id_br_s.iloc[:,-1]


### Vocabulary size
voc_size=9000


messages.reset_index(inplace = True)
#messages['title']

#Download the nltk toolkit
# nltk.download('stopwords')
# nltk.download('punkt')


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
    words=list(Word2VecModel.wv.index_to_key)#index_to_key  vocab
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


MAX_SEQUENCE_LENGTH = 3000
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
summary_description_word2vec_idex=tokenizer(sentences, word_index)
# print(summary_description_word2vec_idex.shape)
# print(lastweight.shape)


main_input = keras.Input(shape=(summary_description_word2vec_idex.shape[1],), name="bugreport")
weight_input = keras.Input(shape=(lastweight.shape[1],), name="weight")
main_features =Embedding(output_dim=embeddings_matrix.shape[1], input_dim=embeddings_matrix.shape[0],weights=[embeddings_matrix], input_length=3000)(main_input)
lstm_out = LSTM(128)(main_features)
x = keras.layers.concatenate([lstm_out, weight_input])
x = Dense(128, activation='relu')(x)
#x = Dense(64, activation='relu')(x)
#x = tf.keras.layers.Flatten(x)
x=tf.keras.layers.Flatten()(x)
output = Dense(1, activation='softmax', name='output')(x)
model = Model(inputs=[main_input, weight_input], outputs=[output])
model.compile(optimizer='rmsprop',
loss={'output': keras.losses.CategoricalCrossentropy(from_logits=True)},
loss_weights={'output': 0.2},
metrics=['accuracy'])
print(model.summary())

# fit
# history=model.fit(
#     {"bugreport": summary_description_word2vec_idex, "weight": lastweight},
#     {"output": y},
#     epochs=10,
#     batch_size=32,
# )



def get_train_batch(source_input_ids,target_input_ids,target_output_ids, batch_size):
    '''
    参数：
        train_dataset:所有数据，为source_data_ids,target_input_ids,target_output_ids, batch_size
        batch_size:批次
    返回:
        一个generator，( inputs = {'encode_input':e,'decode_input':d},outputs =  {'dense':t})
    '''
    while 1:
        for i in range(0, len(source_input_ids), batch_size):
            e = source_input_ids[i:i+batch_size]
            d = target_input_ids[i:i+batch_size]
            t = target_output_ids[i:i+batch_size]
            e = np.array(e)
            d = np.array(d)
            t = np.array(t)
            yield ({'bugreport':e,'weight':d},{'output':t})


batch_size=8;

model.fit(get_train_batch(summary_description_word2vec_idex[:20],lastweight[:20],y_labels[:20], batch_size),
          batch_size=batch_size,
          epochs=1,
          steps_per_epoch = 20,
          verbose=1,
          validation_data=([summary_description_word2vec_idex[20:],lastweight[20:]], y_labels[20:])
)

