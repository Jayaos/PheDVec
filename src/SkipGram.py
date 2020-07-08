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
        self.config = set_config(config_dir)
        self.optimizer = tf.keras.optimizers.Adadelta(self.config.hparams.learning_rate)
        self.concept2id = load_data(self.config.data.concept2id)
        self.training_data = None        
        self.epoch_loss_avg = [] # record avg loss for all epochs

    def loadData(self):
        print("load training data...")
        self.training_data = load_data(self.config.data.training_data)[0]

    def initEmbedding(self):
        print("initialize model...")
        self.embedding = tf.Variable(tf.random.uniform([len(self.concept2id), self.config.hparams.emb_dim], 0.1, -0.1))
    
    @tf.function
    def computeConceptCost(self, i_vec, j_vec):
        logEps = tf.constant(1e-8)
        norms = tf.exp(tf.reduce_sum(tf.matmul(self.embedding, self.embedding, transpose_b=True), axis=1))
        denoms = tf.exp(tf.reduce_sum(tf.multiply(tf.gather(
            self.embedding, i_vec), tf.gather(self.embedding, j_vec)), axis=1))
        emb_cost = tf.negative(tf.math.log(tf.divide(denoms, tf.gather(norms, i_vec))+ logEps))
        
        return tf.reduce_mean(emb_cost)

    def train(self, num_epochs, batch_size):
        cost_avg = tf.keras.metrics.Mean()
        print("start training...")
        for epoch in range(num_epochs):
            total_batch = int(np.ceil(len(self.training_data)) / batch_size)
            progbar = tf.keras.utils.Progbar(total_batch)

            for i in random.sample(range(total_batch), total_batch): # shuffling the data 
                i_vec, j_vec = prepare_batch(self.training_data[batch_size * i:batch_size * (i+1)])
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
                
        self.saveResults(epoch, avg_loss)

    def saveResults(self, epoch, avg_loss):
        print("save trained embedding...")
        np.save(os.path.join(self.config.path.output_path, "phedvec_e{:03d}_loss{:.4f}.npy".format(epoch, avg_loss)),
                np.array(self.embedding[:]))
        print("save avg loss record...")
        save_loss_record(self.epoch_loss_avg, "training_loss_skipgram.txt", self.config.path.output_path)
        
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

def load_data(data_dir):
    with open(data_dir, 'rb') as f:
        my_dict = pickle.load(f)
    return my_dict

def prepare_batch(record):
    i_vec = []
    j_vec = []

    for first in record:
        for second in record:
            if first == second: 
                continue
            i_vec.append(first)
            j_vec.append(second)

    return i_vec, j_vec

def save_loss_record(loss_record, name, save_dir):
    with open(os.path.join(save_dir, name), "w") as f:
        for i in range(len(loss_record)):
            f.write(str(loss_record[i]))
            f.write("\n")