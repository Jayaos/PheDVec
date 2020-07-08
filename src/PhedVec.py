import tensorflow as tf
import json
from dotmap import DotMap
import numpy as np
import pickle
import random
import gensim
from tqdm import tqdm
import os

class PhedVec(tf.keras.Model):
    def __init__(self, config_dir):
        super(PhedVec, self).__init__()
        self.config = set_config(config_dir)
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01)
        self.concept2id = load_data(self.config.data.concept2id)
        self.training_data = None
        self.labels = None
        self.epoch_loss_avg = [] # record avg loss for all epochs
        
        self.visit_activation = tf.keras.layers.Activation(activation=tf.keras.activations.tanh)
        self.phecode_classifier = tf.keras.layers.Dense(self.config.hparams.num_pheclass, 
        name="phe_classifier", activation=tf.keras.activations.softmax) 

    def loadData(self):
        print("load training data...")
        self.training_data, self.labels = load_data(self.config.data.training_data)

    def initModel(self):
        if self.concept2id == None: 
            print("Load concept2id before initialzing the model")

        print("initialize model...")
        self.embedding = tf.Variable(tf.random.uniform([len(self.concept2id), self.config.hparams.emb_dim], 0.1, -0.1))
        
    @tf.function
    def getVisitRep(self, x_batch):
        emb_output = tf.nn.embedding_lookup(self.embedding, x_batch) # n(batch_size) * l(padded_len)
        mask = x_batch != 0
        visit_rep = self.visit_activation(tf.reduce_sum(tf.ragged.boolean_mask(emb_output, mask), axis=1))
        return visit_rep # n(batch_size) * d(dim)
    
    @tf.function
    def computeVisitCost(self, x_batch, label_batch):
        visit_rep = self.getVisitRep(x_batch)
        phecode_prediction = self.phecode_classifier(visit_rep)
        logEps = tf.constant(1e-5)
        visit_cost1 = tf.multiply(label_batch, tf.math.log(tf.math.add(phecode_prediction, logEps)))
        visit_cost2 = tf.multiply(tf.math.subtract(1.0, label_batch), tf.math.log(tf.math.add(tf.math.subtract(1.0,phecode_prediction), logEps)))
        visit_cost = tf.math.divide( tf.math.negative(tf.reduce_sum(tf.math.add(visit_cost1, visit_cost2))), len(x_batch))
        return visit_cost
    
    @tf.function
    def computeConceptCost(self, i_vec, j_vec): 
        logEps = tf.constant(1e-8)
        norms = tf.reduce_sum(tf.math.exp(tf.matmul(self.embedding, self.embedding, transpose_b=True)), axis=1)
        denoms = tf.math.exp(tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(self.embedding, i_vec), 
                                                       tf.nn.embedding_lookup(self.embedding, j_vec)), axis=1))
        concept_cost = tf.negative(tf.math.log((tf.divide(denoms, tf.gather(norms, i_vec)) + logEps)))
        return tf.math.reduce_mean(concept_cost)
    
    def computeTotalCost(self, x_batch, i_vec, j_vec, label_batch):
        batch_cost = tf.math.add(self.computeVisitCost(x_batch, label_batch), self.computeConceptCost(i_vec, j_vec))
        return batch_cost
    
    def saveResults(self, epoch, avg_loss):
        print("save trained embedding...")
        np.save(os.path.join(self.config.path.output_path, "phedvec_e{:03d}_loss{:.4f}.npy".format(epoch, avg_loss)),
                np.array(self.embedding[:]))
        print("save avg loss record...")
        save_loss_record(self.epoch_loss_avg, "training_loss_PhedVec.txt", self.config.path.output_path)

    def train(self, num_epochs, batch_size):
        cost_avg = tf.keras.metrics.Mean()
        print("start training...")
        for epoch in range(num_epochs):
            total_batch = int(np.ceil(len(self.training_data)) / batch_size)
            progbar = tf.keras.utils.Progbar(total_batch)

            for i in random.sample(range(total_batch), total_batch): # shuffling the data 
                x = self.training_data[batch_size * i:batch_size * (i+1)]
                i_vec, j_vec, label = prepare_batch(x, self.labels[batch_size * i:batch_size * (i+1)], self.config.hparams.phecode_num)

                with tf.GradientTape() as tape:
                    batch_cost = self.computeTotalCost(x, i_vec, j_vec, label)
                gradients = tape.gradient(batch_cost, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                cost_avg(batch_cost)
                progbar.add(1)

            if (epoch % 1) == 0: 
                avg_loss = cost_avg.result()
                print("Epoch {}: Loss: {:.4f}".format(epoch+1, avg_loss))
                self.epoch_loss_avg.append(avg_loss.numpy)

        self.saveResults(epoch, avg_loss)

# Functions 
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
    with open(data_dir, "rb") as f:
        mylist = pickle.load(f)
    return mylist

def encode_multihot(labels, category_num):
    multihot_labels = np.zeros((len(labels), category_num))
    for idx, label in enumerate(labels):
        multihot_labels[idx][label] = 1.
    return multihot_labels

def prepare_batch(record, label, category_num):
    i_vec = []
    j_vec = []

    multihot_label = encode_multihot(label, category_num)
    for visit in record:
        pick_two(visit, i_vec, j_vec)

    return np.array(i_vec).astype("int32"), np.array(j_vec).astype("int32"), np.array(multihot_label).astype("float32")

def save_loss_record(loss_record, name, save_dir):
    with open(os.path.join(save_dir, name), "w") as f:
        for i in range(len(loss_record)):
            f.write(str(loss_record[i]))
            f.write("\n")

def pick_two(visit, i_vec, j_vec):
    for first in visit:
        for second in visit:
            if first == second: 
                continue
            i_vec.append(first)
            j_vec.append(second)

    return i_vec, j_vec