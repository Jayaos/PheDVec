import tensorflow as tf
import json
from dotmap import DotMap
import numpy as np
import pickle
import random
import gensim
from tqdm import tqdm
import os

class SkipGram(tf.keras.Model):
    def __init__(self, config_dir):
        super(SkipGram, self).__init__()
        self.config = setConfig(config_dir)
        self.optimizer = tf.keras.optimizers.Adadelta(self.config.hparams.learning_rate)
        self.concept2id = loadDictionary(self.config.data.concept2id)
        self.training_data = None        
        self.epoch_loss_avg = [] # record avg loss for all epochs

    def initEmbedding(self):
        print("initialize model...")
        self.embedding = tf.Variable(tf.random.uniform([len(self.concept2id), 1000], 0.1, -0.1))

    @tf.function
    def computeConceptCost(self, i_vec, j_vec):
        logEps = tf.constant(1e-8)
        norms = tf.exp(tf.reduce_sum(tf.matmul(self.embedding, self.embedding, transpose_b=True), axis=1))
        denoms = tf.exp(tf.reduce_sum(tf.multiply(tf.gather(
            self.embedding, i_vec), tf.gather(self.embedding, j_vec)), axis=1))
        emb_cost = tf.negative(tf.math.log(tf.divide(denoms, tf.gather(norms, i_vec))+ logEps))
        
        return tf.reduce_mean(emb_cost)

    def train(self, num_epochs, batch_size, buffer_size, save_dir):
        cost_avg = tf.keras.metrics.Mean()
        dataset = tf.data.Dataset.from_tensor_slices((self.training_data, self.labels)).shuffle(buffer_size).batch(batch_size)
        for epoch in range(num_epochs):
            total_batch = int(np.ceil(len(self.training_data)) / batch_size)
            progbar = tf.keras.utils.Progbar(total_batch)

            for one_batch in dataset:

                i_vec, j_vec = padMatrix(one_batch)
                with tf.GradientTape() as tape:
                    batch_cost = self.computeConceptCost(i_vec, j_vec)
                gradients = tape.gradient(batch_cost, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                cost_avg(batch_cost)
                progbar.add(1)

            if (epoch % 1) == 0: 
                avg_loss = cost_avg.result()
                print("Epoch {}: Loss: {:.4f}".format(epoch+1, avg_loss))
                self.epoch_loss_avg.append(avg_loss.numpy)
                
        self.saveResults(save_dir, epoch, avg_loss)
        

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

def loadDictionary(data_dir):
    with open(data_dir, 'rb') as f:
        my_dict = pickle.load(f)
    return my_dict

def padMatrix(records):
    i_vec = []
    j_vec = []
    
    for record in records:
        if record[0] != -1:
            pickTwo(record, i_vec, j_vec)

    return i_vec, j_vec

def pickTwo(record, i_vec, j_vec):
    for first in record:
        for second in record:
            if first == second: 
                continue
            i_vec.append(first)
            j_vec.append(second)