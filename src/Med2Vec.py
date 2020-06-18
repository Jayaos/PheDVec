import tensorflow as tf
import json
from dotmap import DotMap
import numpy as np
import pickle
import random
import gensim
from tqdm import tqdm
import os

class Med2Vec(tf.keras.Model):
    def __init__(self, config_dir):
        super(Med2Vec, self).__init__()
        self.config = setConfig(config_dir)
        self.optimizer = tf.keras.optimizers.Adadelta(self.config.hparams.learning_rate)
        self.concept2id = loadDictionary(self.config.data.concept2id)
        self.training_data = None        
        self.visit_activation = tf.keras.layers.Activation(activation=tf.keras.activations.relu)
        self.softmax_prediction = tf.keras.layers.Dense(len(self.concept2id), input_shape=(self.config.hparams.emb_dim,),
        activation=tf.keras.activations.softmax)
        self.epoch_loss_avg = [] # record avg loss for all epochs
        
    def initEmbedding(self):
        print("initialize embedding...")
        self.embedding = tf.Variable(tf.random.uniform([len(self.concept2id), self.config.hparams.emb_dim], 0.1, -0.1))
        self.bias = tf.zeros(self.config.hparams.emb_dim)
        
    @tf.function
    def getPrediction(self, x):
        visit_emb = tf.add(tf.matmul(x, self.embedding), self.bias)
        visit_rep = self.visit_activation(visit_emb)
        softmax_result = self.softmax_prediction(visit_rep)
        return softmax_result


def setConfig(json_file):
    """
    Get the config from a json file
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)
    return config

def pickTwo(record, i_vec, j_vec):
    for first in record:
        for second in record:
            if first == second: 
                continue
            i_vec.append(first)
            j_vec.append(second)

def padMatrix(records, num_codes):
    n_samples = len(records)
    i_vec = []
    j_vec = []
    x = np.zeros((n_samples, num_codes))
    mask = np.zeros((n_samples,))
    
    for idx, record in enumerate(records):
        if record[0] != -1:
            x[idx][record] = 1.
            pickTwo(record, i_vec, j_vec)
            mask[idx] = 1.

    return x.astype("float32"), mask.astype("float32"), i_vec, j_vec

def loadDictionary(data_dir):
    with open(data_dir, 'rb') as f:
        my_dict = pickle.load(f)
    return my_dict