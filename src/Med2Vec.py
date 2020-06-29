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

    def loadData(self):
        print("load training data...")
        self.training_data = load_data(self.config.data.training_data)
        
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
    def computeConceptCost(self, i_vec, j_vec): # serious problem here take a look
        logEps = tf.constant(1e-8)
        preVec = tf.keras.activations.relu(self.embedding)
        norms = tf.reduce_sum(tf.exp(tf.matmul(preVec, preVec, transpose_b=True)), axis=1)
        denoms = tf.exp(tf.reduce_sum(tf.multiply(tf.gather(preVec, i_vec), tf.gather(preVec, j_vec)), axis=1))
        emb_cost = tf.negative(tf.math.log(tf.divide(denoms, tf.gather(norms, i_vec))+ logEps))
        
        return tf.reduce_mean(emb_cost)

    def train(self, num_epochs, batch_size, buffer_size):
        cost_avg = tf.keras.metrics.Mean()
        print("build tensorflow dataset...")
        dataset = tf.data.Dataset.from_tensor_slices(self.training_data).shuffle(buffer_size).batch(batch_size)
        print("start training...")
        for epoch in range(num_epochs):
            total_batch = int(np.ceil(len(self.training_data)) / batch_size)
            progbar = tf.keras.utils.Progbar(total_batch)

            for one_batch in dataset:

                x, mask, i_vec, j_vec = padMatrix(one_batch, len(self.concept2id))

                with tf.GradientTape() as tape:
                    batch_cost = tf.add(self.computeConceptCost(i_vec, j_vec), 
                    self.computeVisitCost(x, mask))
                gradients = tape.gradient(batch_cost, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                cost_avg(batch_cost)
                progbar.add(1)

            if (epoch % 1) == 0: 
                avg_loss = cost_avg.result()
                print("Epoch {}: Loss: {:.4f}".format(epoch+1, avg_loss))
                self.epoch_loss_avg.append(avg_loss.numpy)
                
        self.saveResults()
    
    def saveResults(self):
        print("save trained embedding...")
        save_variable(self.embedding, "med2vec_emb.npy", self.config.dir.save_dir)
        print("save avg loss record...")
        save_loss_record(self.epoch_loss_avg, "training_loss_record.txt", self.config.dir.save_dir)


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

def load_data(data_dir):
    with open(data_dir, "rb") as f:
        mylist = pickle.load(f)
    return mylist

def save_variable(variable_matrix, name, save_dir):
    np.save(os.path.join(save_dir, name), variable_matrix)

def save_loss_record(loss_record, name, save_dir):
    with open(os.path.join(save_dir, name), "w") as f:
        for i in range(len(loss_record)):
            f.write(loss_record[i])
            f.write("\n")
        
