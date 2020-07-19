import numpy as np
import tensorflow as tf
from tqdm import tqdm
import itertools
import random
import sys
import os
import pickle
from dotmap import DotMap
import json
from collections import defaultdict

class GloVe(tf.keras.Model):
    def __init__(self, json_dir, max_vocab_size=100, scaling_factor=0.75):
        super(GloVe, self).__init__()
        self.config = set_config(json_dir)

        self.max_vocab_size = max_vocab_size
        self.scaling_factor = scaling_factor
        self.training_data = None        
        self.concept2id = load_dictionary(self.config.data.concept2id)
        self.vocab_size = len(self.concept2id)
        self.comap = None
        self.comatrix = None
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.config.hparams.learning_rate)
        self.epoch_loss_avg = []

    def loadData(self):
        print("load training data...")
        self.training_data = load_data(self.config.data.training_data_phedvec)[0]
     
    def buildCoMatrix(self):
        self.comatrix = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)

        for i in tqdm(range(len(self.training_data))):
            record = unpad_list(self.training_data[i])
            for p in record:
                for k in record:
                    if p != k:
                        self.comatrix[p, k] += 1.

    def initParams(self):
        with tf.device("/cpu:0"):
            """must be implemented with cpu-only env since this is sparse updating"""
            self.target_embeddings = tf.Variable(tf.random.uniform([self.vocab_size, self.config.hparams.emb_dim], 0.1, -0.1),
                                                 name="target_embeddings")
            self.context_embeddings = tf.Variable(tf.random.uniform([self.vocab_size, self.config.hparams.emb_dim], 0.1, -0.1),
                                                  name="context_embeddings")
            self.target_biases = tf.Variable(tf.random.uniform([self.vocab_size], 0.1, -0.1),
                                             name='target_biases')
            self.context_biases = tf.Variable(tf.random.uniform([self.vocab_size], 0.1, -0.1),
                                              name="context_biases")

    def computeCost(self, x):
        with tf.device("/gpu:0"):
            """x = [target_ind, context_ind, co_occurrence_count]"""
            target_emb = tf.nn.embedding_lookup([self.target_embeddings], x[0])
            context_emb = tf.nn.embedding_lookup([self.context_embeddings], x[1])
            target_bias = tf.nn.embedding_lookup([self.target_biases], x[0])
            context_bias = tf.nn.embedding_lookup([self.context_biases], x[1])

            weight = tf.math.minimum(1.0, tf.math.pow(tf.math.truediv(x[2], tf.cast(self.max_vocab_size, dtype=tf.float32)),
                                                         self.scaling_factor))
        
            emb_product = tf.math.reduce_sum(tf.math.multiply(target_emb, context_emb), axis=1)
            log_cooccurrence = tf.math.log(tf.add(tf.cast(x[2], dtype=tf.float32), 1))
        
            distance_cost = tf.math.square(
                tf.math.add_n([emb_product, target_bias, context_bias, tf.math.negative(log_cooccurrence)]))
               
            batch_cost = tf.math.reduce_sum(tf.multiply(weight, distance_cost))
          
        return batch_cost

    def computeGradients(self, x):
        with tf.GradientTape() as tape:
            cost = self.computeCost(x)
        return cost, tape.gradient(cost, self.trainable_variables)

    def getEmbeddings(self):
        self.embeddings = self.target_embeddings + self.context_embeddings
    
    def saveEmbeddings(self, epoch, avg_loss):
        self.getEmbeddings()
        np.save(os.path.join(self.config.path.output_path, "glove_emb_e{:03d}_loss{:.4f}.npy".format(epoch, avg_loss)),
                self.embeddings)
        print("Embedding results have been saved")

    def train(self, num_epochs, batch_size):
        i_ids, j_ids, co_occurs = prepare_trainingset(self.comatrix)
        total_batch = int(np.ceil(len(i_ids) / batch_size))
        cost_avg = tf.keras.metrics.Mean()

        for epoch in range(num_epochs):
            progbar = tf.keras.utils.Progbar(len(i_ids))
            
            for i in random.sample(range(total_batch), total_batch): # shuffling the data 
                i_batch = i_ids[i * batch_size : (i+1) * batch_size]
                j_batch = j_ids[i * batch_size : (i+1) * batch_size]
                co_occurs_batch = co_occurs[i * batch_size : (i+1) * batch_size]
                cost, gradients = self.computeGradients([i_batch, j_batch, co_occurs_batch])
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                cost_avg(cost) 
                progbar.add(batch_size)
                print("Step {}: Loss: {:.4f}".format(self.optimizer.iterations.numpy(), cost))
                
            if (epoch % 1) == 0: 
                avg_loss = cost_avg.result()
                print("Epoch {}: Loss: {:.4f}".format(epoch, avg_loss))
                self.epoch_loss_avg.append(avg_loss)
                    
        self.saveEmbeddings(epoch, avg_loss)

def set_config(json_file):
    """
    Get the config from a json file
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)
    return config

def load_dictionary(data_dir):
    with open(data_dir, 'rb') as f:
        my_dict = pickle.load(f)
    return my_dict

def prepare_trainingset(comatrix):
    i_ids = []
    j_ids = []
    co_occurs = []

    for i in range(comatrix.shape[0]):
        for j in range(comatrix.shape[0]):
            i_ids.append(i)
            j_ids.append(j)
            co_occurs.append(comatrix[i, j])
     
    assert len(i_ids) == len(j_ids), "The length of the data are not the same"
    assert len(i_ids) == len(co_occurs), "The length of the data are not the same"
    return i_ids, j_ids, co_occurs

def load_data(data_dir):
    with open(data_dir, "rb") as f:
        mylist = pickle.load(f)
    return mylist

def unpad_list(mylist):
    padding_ind = len(mylist)
    for i in range(len(mylist)):
        if mylist[i] == 0:
            padding_ind = i
            break
    return mylist[:padding_ind]