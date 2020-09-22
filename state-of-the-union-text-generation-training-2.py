import os, sys
import pathlib

## --- config ----
## --- end config


DATA_FILES_CLINTON = ['data/1993-Clinton.txt', 'data/1994-Clinton.txt', 'data/1995-Clinton.txt', 'data/1996-Clinton.txt',
                      'data/1997-Clinton.txt', 'data/1998-Clinton.txt', 'data/1999-Clinton.txt']

DATA_FILES_GWBUSH = ['data/2001-GWBush-1.txt', 'data/2001-GWBush-2.txt', 'data/2002-GWBush.txt', 'data/2003-GWBush.txt', 'data/2004-GWBush.txt',
                    'data/2005-GWBush.txt', 'data/2006-GWBush.txt', 'data/2007-GWBush.txt', 'data/2008-GWBush.txt']

DATA_FILES_OBAMA = ['data/2009-Obama.txt', 'data/2010-Obama.txt', 'data/2011-Obama.txt', 'data/2012-Obama.txt', 
              'data/2013-Obama.txt', 'data/2014-Obama.txt', 'data/2015-Obama.txt', 'data/2016-Obama.txt']

DATA_FILES_TRUMP = ['data/2017-Trump.txt', 'data/2018-Trump.txt', 'data/2019-Trump.txt', 'data/2020-Trump.txt']

DATA_FILES_ALL = DATA_FILES_CLINTON + DATA_FILES_GWBUSH + DATA_FILES_OBAMA + DATA_FILES_TRUMP


## ======== tensorflow init ===========
## disable info logs from TF
#   Level | Level for Humans | Level Description                  
#  -------|------------------|------------------------------------ 
#   0     | DEBUG            | [Default] Print all messages       
#   1     | INFO             | Filter out INFO messages           
#   2     | WARNING          | Filter out INFO & WARNING messages 
#   3     | ERROR            | Filter out all messages 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel('WARN')

print ('tensorflow version :', tf.__version__)
print(tf.config.experimental.list_physical_devices())


# ## TF-GPU Debug
# The following block tests if TF is running on GPU.


## This block is to tweak TF running on GPU
## You may comment this out, if you are not using GPU

## ---- start Memory setting ----
## Ask TF not to allocate all GPU memory at once.. allocate as needed
## Without this the execution will fail with "failed to initialize algorithm" error

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
## ---- end Memory setting ----


models_to_build =  [
    {
        'model_name' :   'sotu-clinton',
        'data_files' : DATA_FILES_CLINTON
    },
    {
        'model_name' :   'sotu-gwbush',
        'data_files' : DATA_FILES_GWBUSH
    },
    {
        'model_name' :   'sotu-obama',
        'data_files' : DATA_FILES_OBAMA
    },
    {
        'model_name' :   'sotu-trump',
        'data_files' : DATA_FILES_TRUMP
    },
    {
        'model_name' :   'sotu-last4',
        'data_files' : DATA_FILES_ALL
    },
]


def get_corpus(data_files, model_name):
    print ("data files: ", data_files)

    corpus = []
    for file_name in data_files:
        with open(file_name, 'r') as file:
            file_contents = file.readlines()
            # print ('file : ', file_name, ', num lines : ', len(file_contents))
            
            for line in file_contents:
                corpus.append(line)  

    # print ('corpus total num lines : ', len(corpus))


    # ## Step - Text Cleanup
    # 
    # - lower case all text
    # - remove punctuations
    # - We are not removing stop words, as it may impact the meaning of text


    from nltk.tokenize import RegexpTokenizer

    corpus_clean = []

    # this will tokenize full words, seperate from punctuations
    regex_tokenizer = RegexpTokenizer(r'\w+')

    word_count_corpus = 0
    word_count_corpus_clean = 0
    for sentence in corpus:
        #print (sentence)
        
        ## step 1 : lowercase
        sentence = sentence.lower()
        
        ##  Step 2 : break into words
        words = regex_tokenizer.tokenize (sentence)
        word_count_corpus += len(words)
        
        ## Step 3 : remove punctuations, numbers ..etc
        words_clean =[word for word in words if word.isalpha()]
        word_count_corpus_clean += len(words_clean)
        
        #print ("words:" , words)
        
        if len(words_clean) > 0:
            sentence_clean = " ".join(words_clean)
            #print (sentence_clean)
            #print ("====")
            corpus_clean.append(sentence_clean)
        
    # print ('{}: word_count_corpus : {:,}'.format(model_name, word_count_corpus))
    # print ('{}: word_count_corpus_clean : {:,}'.format(model_name, word_count_corpus_clean))
    # print ('{}: removed words : {:,}'.format(model_name, (word_count_corpus - word_count_corpus_clean)))

    return corpus_clean, word_count_corpus_clean
# ===== end: get_cropus ============

def sample_from_dict(d, sample=10):
    import random
    
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))
## ==== end : sample_from_dict(d, sample=10):

def tokenize_corpus(model_name, corpus):
    from tensorflow.keras.preprocessing.text import Tokenizer
    import os

    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(corpus)

    ## Basic info
    from  collections import Counter
    from pprint import pprint

    print ('number of  unique words : {:,}'.format(len(tokenizer.word_index)+1))
    #print ('\nSome random word mappings : ')
    #pprint (sample_from_dict(tokenizer.word_index))

    counter = Counter(tokenizer.word_counts)
    #print ('\nTop-N words:')
    #pprint(counter.most_common(20))


    # Step - Save Tokenizer
    import json 
    tokenizer_json = tokenizer.to_json()
    tok_file_output = os.path.join("tokenizer-vocabulary", model_name + '-vocab.json')
    os.makedirs(os.path.dirname(tok_file_output), exist_ok=True)
    with open(tok_file_output, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    print ("Saved tokenizer vocab to : ", tok_file_output)

    return tokenizer
## === end:  tokenize_corpus(model_name, corpus):

## Create Input Sequences
# 
# Here we are creating ngram sequence like this:
# 
# ```text
# line:  mr. speaker, mr. vice president, members of congress, my fellow
# 
#    token_list:  [505, 533, 505, 534, 206, 506, 4, 77, 84, 423]
#      1: n_gram_sequence: [505, 533]
#      1: input_sequences: [[505, 533]]
# 
#      2: n_gram_sequence: [505, 533, 505]
#      2: input_sequences: [[505, 533], [505, 533, 505]]
# 
#      3: n_gram_sequence: [505, 533, 505, 534]
#      3: input_sequences: [[505, 533], [505, 533, 505], [505, 533, 505, 534]]
# 
#      4: n_gram_sequence: [505, 533, 505, 534, 206]
#      4: input_sequences: [[505, 533], [505, 533, 505], [505, 533, 505, 534], [505, 533, 505, 534, 206]]
# ```
def create_sequence(corpus, tokenizer):
    ## for debug, uncomment this and test it on smaller corpus
    # corpus_clean = corpus_clean[:10]

    input_sequences = []

    for line in corpus:
        #print ('line: ', line)
        token_list = tokenizer.texts_to_sequences([line])[0]
        #print ('   token_list: ', token_list)
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            #print ('     {}: n_gram_sequence: {}'.format(i, n_gram_sequence))
            input_sequences.append(n_gram_sequence)
            #print ('     {}: input_sequences: {}'.format(i,input_sequences ))
            #print()

    total_words = len(tokenizer.word_index) + 1

    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # pad sequences 
    max_sequence_len = max([len(x) for x in input_sequences])
    print ('max_sequence_len: ', max_sequence_len)

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    # print ('\ninput_sequences:')
    # print(input_sequences)

    # create predictors and label
    xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
    # print ('\nxs:')
    # print (xs)
    # print ('\nlabels')
    # print (labels)

    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    # print ('\nys')
    # print (ys)

    print ('xs.shape :', xs.shape)
    print ('ys.shape :', ys.shape)

    return input_sequences, max_sequence_len, xs, ys
# === end:def create_sequence(model_name, tokenizer):

def setup_tensorboard (model_name):
    import datetime
    import os
    import shutil

    # timestamp  = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    tb_top_level_dir= '/tmp/tensorboard-logs'
    tb_app_dir = os.path.join (tb_top_level_dir, model_name)
    tb_logs_dir = os.path.join (tb_app_dir, datetime.datetime.now().strftime("%H-%M-%S"))

    print ("Saving TB logs to : " , tb_logs_dir)
    #clear out old logs
    shutil.rmtree ( tb_app_dir, ignore_errors=True )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_logs_dir, write_graph=True, 
                                                        write_images=True, histogram_freq=1)
    return tensorboard_callback
# ======  end: setup_tensorboard (model_name):

def build_model(num_unique_words, max_sequence_len):
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam


    ## model 1
    # model_version = "1"
    # model = Sequential([
    #             Embedding(input_dim=num_unique_words, output_dim=100, input_length=max_sequence_len-1),
    #             Bidirectional(LSTM(64)),
    #             Dense(num_unique_words, activation='softmax')
    #     ])

    ## Model 2
    model_version = "2"
    model = Sequential([
                Embedding(input_dim=num_unique_words, output_dim=100, input_length=max_sequence_len-1),
                Bidirectional(LSTM(64, return_sequences=True)),
                Bidirectional(LSTM(64)),
                Dense(num_unique_words, activation='softmax')
        ])

    ## Model 3
    # model_version = "3"
    # model = Sequential([
    #             Embedding(input_dim=num_unique_words, output_dim=100, input_length=max_sequence_len-1),
    #             Bidirectional(LSTM(128, return_sequences=True)),
    #             Dropout(0.2),
    #             Bidirectional(LSTM(128)),
    #             Dense(128, activation='relu'),
    #             Dense(num_unique_words, activation='softmax')
    #     ])

    model.compile(loss='categorical_crossentropy', 
                optimizer = 'adam',
                # optimizer = Adam(lr=0.01)
                metrics=['accuracy'])
        
    model.summary()

    # tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    return model, model_version
# === end : build_model():

def train_model(model_name, model_version, model, xs, ys):
    import time
    import os
    import humanize
    import datetime as dt

    cb_tensorboard = setup_tensorboard(model_name + "-" + model_version)

    cb_early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.01, patience=10, verbose=1)

    checkpoint_path = os.path.join("model-checkpoints" , model_name ,  model_version,  "/model.ckpt")
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    ## train with validation
    t1 = time.perf_counter()
    history = model.fit(xs, ys, validation_split=0.2, 
                        epochs=500, verbose=2, 
                        callbacks=[cb_tensorboard, cb_early_stop])
    t2 = time.perf_counter()

    print ("*** training done in {:,.1f} seconds or  {}".format ((t2-t1), 
                humanize.time.precisedelta(dt.timedelta(seconds=t2-t1))))

    ## Save model
    import os
    model_file = os.path.join('models', model_name + '-model-' + model_version + '.h5')
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    model.save(model_file)

    model_size_in_bytes = os.path.getsize(model_file)
    print ("model saved as '{}',  size = {:,} bytes / {:,.1f} KB  / {:,.1f} MB".format(model_file, 
                                        model_size_in_bytes, model_size_in_bytes / 1024, 
                                        model_size_in_bytes / (1024*1024) ))

    return history
# === end train_model(model):

def plot_learning_curve(model_name, model_version, history):
    import matplotlib.pyplot as plt
    import os

    graph_file = os.path.join('plots', model_name + '-' + model_version + ".png")
    os.makedirs(os.path.dirname(graph_file), exist_ok=True)

    plt.plot(history.history['accuracy'], label='train_accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.grid(True)
    plt.title(model_name + '-' + model_version)

    plt.savefig(graph_file)
    plt.clf()  # reset 
    plt.close()
    print ('Learning curve saved to : ', graph_file)
# === end :  plot_learning_curve



for model_to_build in models_to_build:
    model_name = model_to_build['model_name']
    print ('=============== building model : ', model_name , ' ==============')
    corpus, num_words = get_corpus(model_to_build['data_files'], model_name)
    print ('got corpus with word count : {:,}'.format (num_words))

    tokenizer  = tokenize_corpus(model_name,  corpus)
    num_unique_words = len(tokenizer.word_index) + 1

    input_seqeuences, max_sequence_len, xs, ys = create_sequence(corpus, tokenizer)

    model, model_version = build_model(num_unique_words, max_sequence_len)

    history = train_model(model_name, model_version, model, xs, ys)

    plot_learning_curve(model_name, model_version, history)
# --- end for loop