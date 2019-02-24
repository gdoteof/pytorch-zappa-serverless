from flask import Flask
from flask_cors import CORS, cross_origin
from flask import Response
import flask
import json
import pickle
import os
import requests
# from fastai.text import get_language_model
from pytorch_models import awd_lstm
import numpy as np
import torch 
import boto3
import os.path

from io import BytesIO
import urllib.request

from fastai.vision import *

app = Flask(__name__)
CORS(app)

def maybe_fetch_s3(bucket, path):
    filename=f'/tmp/{path}'
    if not os.path.exists(filename):
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, path)
        obj.get_contents_to_filename(filename)
    return filename
    
def sample_model(model, input_words, l=50):
    '''
    model: pytorch LM 
    s: list of strings
    '''
    no_space_words = ["'s", "'ll", ",", "?",".", "'t", "'m", "n't", "!", "'", "'ve", ";", "http", ":", "/", "\\"]
    capitalize_words = ['.', '!', '\n']
    exclude_tokens = [model.stoi[i] for i in ["xxup", "xxfld", "xxrep"] if i in model.stoi]
    bs = model[0].bs
    model[0].bs=1
    model.eval()
    model.reset()
    final_string = ''
    # Gives the model the input strings
    for s in input_words:
        t = model.stoi[s] if s in model.stoi else 0 
        res,*_ = model.forward(torch.tensor([[t]]).cpu())
        final_string = final_string + ' ' + s
    last_word = None
    # predicts l number of next words
    for i in range(l):
        result_indexes = torch.multinomial(res[-1].exp(), 10)
        # selects a word that is not in the exclude_tokens list
        for r in result_indexes:
            if r != 0 and r not in exclude_tokens: 
                word_index = r
                break
            else:
                word_index = result_indexes[0]
        
        word = model.itos[word_index]
        res, *_ = model.forward(torch.tensor([[word_index.tolist()]]).cpu())
        # Capitalize if it is the last word in a phrase
        word = word.capitalize() if last_word in  capitalize_words else word
        # Do not place a space in front of the words in no_space_words
        if word in no_space_words:
            final_string = final_string + word
        else: 
            final_string = final_string + ' ' + word
        last_word = word
    model[0].bs=bs
    return final_string


def load_lm_and_predict(url):
    ''' input_data: data to be used in the prediction
    model_path: path to the 
    '''
    MODEL_BUCKET=os.environ["models_bucket"]
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    bytes = response.read()

    img = open_image(BytesIO(bytes))
    learner = load_learner(Path("."))
    _,_,losses = learner.predict(img)
    return {
        "predictions": sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True

        )}

@app.route('/which-mouse',methods=['GET'])
def inference():
    ''' 
    GET: Performs inference on the language model
    '''
    response = load_lm_and_predict('https://upload.wikimedia.org/wikipedia/en/d/d4/Mickey_Mouse.png')
    resp = Response(response=json.dumps({"response": response}), status=200, mimetype='application/json')
    return resp


IS_LOCAL = False
if __name__ == '__main__':
    env = 'dev'
    # Test if the values were set to know if it is running locally or on lambda
    json_data = open('zappa_settings.json')
    env_vars = json.load(json_data)
    for key, val in env_vars[env]['aws_environment_variables'].items():
        os.environ[key] = val
    print('Set the environ')
    app.run(debug=True, host='0.0.0.0', port=8082)
