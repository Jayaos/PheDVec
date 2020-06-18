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

    @tf.function
    def computeVisitCost(self, x, mask):
        logEps = tf.constant(1e-8)
        pred = self.getPrediction(x)
        mask_1 = (mask[:-1] * mask[1:])[:, None]

        forward_results =  tf.multiply(pred[:-1], mask_1)
        forward_cross_entropy = -tf.add(tf.multiply(x[1:], tf.math.log(forward_results + logEps)),
        tf.multiply((1. - x[1:]), tf.math.log(1. - forward_results + logEps)))
        backward_results =  tf.multiply(pred[1:], mask_1)
        backward_cross_entropy = -tf.add(tf.multiply(x[:-1], tf.math.log(backward_results + logEps)),
        tf.multiply((1. - x[:-1]), tf.math.log(1. - backward_results + logEps)))

        visit_cost = tf.divide(tf.add(tf.reduce_sum(forward_cross_entropy, axis=[0, 1]), tf.reduce_sum(backward_cross_entropy, axis=[0, 1])), 
        tf.add(tf.reduce_sum(mask_1), logEps))

        return visit_cost

    @tf.function
    def computeConceptCost(self, i_vec, j_vec):
        logEps = tf.constant(1e-8)
        preVec = tf.maximum(self.embedding, 0)
        norms = tf.exp(tf.reduce_sum(tf.matmul(preVec, preVec, transpose_b=True), axis=1))
        denoms = tf.exp(tf.reduce_sum(tf.multiply(tf.gather(preVec, i_vec), tf.gather(preVec, j_vec)), axis=1))
        emb_cost = tf.negative(tf.math.log(tf.divide(denoms, tf.gather(norms, i_vec))+ logEps))
        
        return tf.reduce_mean(emb_cost)
        
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