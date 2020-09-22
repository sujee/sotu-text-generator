import os, sys


## ---config ----

SEED_TEXT_MAX_LENGTH = 50
NUM_WORDS_MAX = 50

models_info = [
    {
        'model_name' : 'clinton-1',
        'model_file' : 'models/sotu-clinton-model-1.h5',
        'vocab_file' : 'models/sotu-clinton-vocab.json',
        'max_sequence_len' : 284
    },
    # {
    #     'model_name' : 'clinton-2',
    #     'model_file' : 'models/sotu-clinton-model-2.h5',
    #     'vocab_file' : 'models/sotu-clinton-vocab.json',
    #     'max_sequence_len' : 284
    # },
    # {
    #     'model_name' : 'clinton-3',
    #     'model_file' : 'models/sotu-clinton-model-3.h5',
    #     'vocab_file' : 'models/sotu-clinton-vocab.json',
    #     'max_sequence_len' : 284
    # },
    {
        'model_name' : 'gwbush-1',
        'model_file' : 'models/sotu-gwbush-model-1.h5',
        'vocab_file' : 'models/sotu-gwbush-vocab.json',
        'max_sequence_len' : 182
    },
    # {
    #     'model_name' : 'gwbush-2',
    #     'model_file' : 'models/sotu-gwbush-model-2.h5',
    #     'vocab_file' : 'models/sotu-gwbush-vocab.json',
    #     'max_sequence_len' : 182
    # },
    # {
    #     'model_name' : 'gwbush-3',
    #     'model_file' : 'models/sotu-gwbush-model-3.h5',
    #     'vocab_file' : 'models/sotu-gwbush-vocab.json',
    #     'max_sequence_len' : 182
    # },
    {
        'model_name' : 'obama-1',
        'model_file' : 'models/sotu-obama-model-1.h5',
        'vocab_file' : 'models/sotu-obama-vocab.json',
        'max_sequence_len' : 132,
    },
    # {
    #     'model_name' : 'obama-2',
    #     'model_file' : 'models/sotu-obama-model-2.h5',
    #     'vocab_file' : 'models/sotu-obama-vocab.json',
    #     'max_sequence_len' : 132,
    # },
    # {
    #     'model_name' : 'obama-3',
    #     'model_file' : 'models/sotu-obama-model-3.h5',
    #     'vocab_file' : 'models/sotu-obama-vocab.json',
    #     'max_sequence_len' : 132,
    # },
    {
        'model_name' : 'trump-1',
        'model_file' : 'models/sotu-trump-model-1.h5',
        'vocab_file' : 'models/sotu-trump-vocab.json',
        'max_sequence_len' : 159
    },
    # {
    #     'model_name' : 'trump-2',
    #     'model_file' : 'models/sotu-trump-model-2.h5',
    #     'vocab_file' : 'models/sotu-trump-vocab.json',
    #     'max_sequence_len' : 159
    # },
    # {
    #     'model_name' : 'trump-3',
    #     'model_file' : 'models/sotu-trump-model-3.h5',
    #     'vocab_file' : 'models/sotu-trump-vocab.json',
    #     'max_sequence_len' : 159
    # },
    {
        'model_name' : 'last4-1',
        'model_file' : 'models/sotu-last4-model-1.h5',
        'vocab_file' : 'models/sotu-last4-vocab.json',
        'max_sequence_len' : 284
    },
    # {
    #     'model_name' : 'last4-2',
    #     'model_file' : 'models/sotu-last4-model-2.h5',
    #     'vocab_file' : 'models/sotu-last4-vocab.json',
    #     'max_sequence_len' : 284
    # },
    # {
    #     'model_name' : 'last4-3',
    #     'model_file' : 'models/sotu-last4-model-3.h5',
    #     'vocab_file' : 'models/sotu-last4-vocab.json',
    #     'max_sequence_len' : 284
    # },

]

## --- end config ----

from flask import Flask, render_template,  Response, request, url_for, flash, redirect
from flask_bootstrap import Bootstrap
import pprint
import logging
from logging.config import dictConfig
import time

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s.%(msecs)03d, %(levelname)s, %(message)s',
            'datefmt': '%Y-%m-%dT%H:%M:%S'
        },
    },
    'handlers': {
        'stdout': {
            'class': "logging.StreamHandler",
            'stream': 'ext://sys.stdout',
            'formatter': 'default'
        }
    },
    'root': {
        'handlers': ['stdout'],
        'level': os.getenv('APP_LOG_LEVEL', 'INFO')},
})

# instantiate flask 
app = Flask(__name__)
bootstrap = Bootstrap(app)
app.logger.setLevel(logging.INFO)
logging.basicConfig(filename='app.log', level=logging.INFO)
app.logger.info("App started")


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
tf.get_logger().setLevel('WARN')

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
## ======== end tensorflow init ===========

from tensorflow.keras.preprocessing.text import tokenizer_from_json

import json 

## ---- init ------

## Load models + data
models_loaded = [] # clean

for model_info in models_info:
    app.logger.info ('Trying to load model: %s', model_info['model_name'])

    if not os.path.isfile(model_info['model_file']):
        app.logger.info("  Model file '%s' is not found.  skipping", model_info['model_file'])
        continue
    if not os.path.isfile(model_info['vocab_file']):
        app.logger.info("  Vocab file '%s' is not found.  skipping", model_info['vocab_file'])
        continue


    ## Load tokenizer
    with open(model_info['vocab_file']) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        model_info['tokenizer'] = tokenizer
        app.logger.info ('   tokenizer vocab loaded from : %s', model_info['vocab_file'])

        ## Create reverse Index for lookup
        model_info['word2index'] = tokenizer.word_index
        model_info['index2word'] = {v:k for (k,v) in tokenizer.word_index.items()}
    # --- end width


    ## -- load models---
    from tensorflow.keras.models import load_model
    model_size_in_bytes = os.path.getsize(model_info['model_file'])
    model = load_model(model_info['model_file'])
    msg =  ("Loaded model from '{}',  size = {:,.1f} MB".format(model_info['model_file'], model_size_in_bytes / (1024*1024) ))
    app.logger.info ("   %s", msg)
    model_info['model'] = model
    model_info['model_size_bytes'] = model_size_in_bytes

    models_loaded.append(model_info)

# --- end for loop

msg = pprint.pformat(models_loaded, depth=2, indent=4)
#app.logger.debug(msg)

## ---- end init -----


### --------  functions ----------
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

#def generate_internal(seed_text, num_words, model_name, model, tokenizer, max_sequence_len, index2word):
def generate_internal(seed_text, num_words, model_info):

    global app

    tokenizer = model_info['tokenizer']
    max_sequence_len = model_info['max_sequence_len']
    index2word = model_info['index2word']
    model = model_info['model']
    model_name = model_info['model_name']

    #app.logger.info("Generating text using model : %s", model_name)

    text = seed_text
    t2a = time.perf_counter()
    for i in range(num_words):
        #print ('{} {} input text : {}'.format(model_name, i,text))
        token_list = tokenizer.texts_to_sequences([text])[0]
        #print ('{} {} token_list: {}'.format(model_name, i, token_list))
        word_list = []
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        #print ('{} token_list padded: {}'.format(i, token_list))
        
        t1a = time.perf_counter()
        prediction_softmax = model.predict(token_list, verbose=0)
        t1b = time.perf_counter()
        #print ("#### prediction in {:,.2f} ms".format((t1b-t1a)*1e3))
        predicted_idx = [ np.argmax(p) for p in prediction_softmax][0]
        
        #print ('{} {} predicted_idx : {}'.format(model_name, i, predicted_idx))
        output_word = index2word.get(predicted_idx, "<UNK>")
        #print ('{} {} output_word : {}'.format(model_name, i, output_word))
        text += " " + output_word
        #print ('{} {} output_text: {}'.format (model_name, i, text))
        #print()
    # --- end for loop

    t2b = time.perf_counter()
    #print ("#### prediction in {:,.2f} ms".format((t2b-t2a)*1e3))
    msg = "   model {}: generated text (in {:,.0f} ms):  {}".format(model_name, (t2b-t2a)*1e3, text)
    app.logger.info(msg)
    return (text)


def generate_text (seed_text, num_words):
    generated_texts = []
    for model_info in models_loaded:
        t1a = time.perf_counter()
        gen_text = generate_internal(seed_text, num_words, model_info)
        t1b = time.perf_counter()
        #print ("#### generate_text_internal in {:,.2f} ms".format((t1b-t1a)*1e3))
        idx = gen_text.index(seed_text)
        gen_text_info = {
            'model_name' : model_info['model_name'],
            'seed_text' : seed_text, 
            'num_words' : num_words,
            'generated_text' : gen_text,
            'generated_text2' : gen_text[idx + len(seed_text) : ]
        }
        generated_texts.append(gen_text_info)
    # --- end for model in models:
    return  generated_texts
# ---- end: def generate_text (seed_text, num_words):


### --------  end functions ----------



@app.route('/', methods=('GET', 'POST'))
def index():

    try:
        seed_text = ''
        num_words_default = 20

        if request.method == 'POST':
            pprint.pprint (request.form)
            seed_text = request.form['seed_text']
            num_words = request.form['num_words']

            try:
                num_words = int(num_words)
            except ValueError:
                num_words = num_words_default

            # lowercase the seed text
            seed_text = seed_text.lower()

            warnings = []
            ## put some limits in
            if len(seed_text) > SEED_TEXT_MAX_LENGTH:
                seed_text = seed_text[:SEED_TEXT_MAX_LENGTH]
                warnings.append('Seed text truncated to ' + str(SEED_TEXT_MAX_LENGTH))

            if  num_words > NUM_WORDS_MAX:
                num_words = NUM_WORDS_MAX
                warnings.append('num_words capped at ' + str(NUM_WORDS_MAX))

            # TODO save it form
            #request.form['seed_text'] = seed_text
            #request.form['num_words'] = num_words

            # if all good, then go ahead
            t1a = time.perf_counter()
            generated_text_info = generate_text(seed_text, num_words)
            t1b = time.perf_counter()
            #print ("#### text generated in {:,.2f} ms".format((t1b-t1a)*1e3))

            msg = pprint.pformat(generated_text_info, depth=2, indent=4)
            app.logger.debug(msg)

            params = {'seed_text' : seed_text, 'num_words' : num_words}

            return (render_template("index.html", 
                    time_took = "{:,.2f}".format(t1b-t1a),
                    warnings = warnings,
                    params = params,
                    generated_text_info = generated_text_info
                    ))

        return (render_template("index.html"))

    except Exception as e:
        print (str(e))
        ## TODO render index template with error



if __name__ == "__main__":
    app.run()