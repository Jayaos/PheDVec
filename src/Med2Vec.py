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
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01)
        self.concept2id = None
        self.training_data = None
        self.labels = None
        self.epoch_loss_avg = [] # record avg loss for all epochs
        
        self.visit_activation = tf.keras.layers.Activation(activation=tf.keras.activations.relu)

    def fitToData(self):
        with tf.device("/cpu:0"):
            patient_data= readPatientRecord(self.config.data.patient_record)
            unique_concepts = getUniqueSet(patient_data)
            self.concept2id = buildDict(list(unique_concepts))
            self.training_data = processPatientRecord(patient_data, self.concept2id)
        print("fitting process has been completed")

    def initModel(self):
        if self.concept2id == None: 
            print("set concept2id before initialzing the model")

        print("initialize model...")
        self.embedding = tf.Variable(tf.random.uniform([len(self.concept2id)+1, 1000], 0.1, -0.1))

    @tf.function
    def getVisitRep(self, x_batch):
        emb_output = tf.nn.embedding_lookup(self.embedding, x_batch) # n(batch_size) * l(padded_len)
        mask = x_batch != 0
        visit_rep = self.visit_activation(tf.reduce_sum(tf.ragged.boolean_mask(emb_output, mask), axis=1))
        return visit_rep # n(batch_size) * d(dim)

    @tf.function
    def computeVisitCost(self, x_batch, mask):
        visit_rep = self.getVisitRep(x_batch)
        mask = (mask[:-1] * mask[1:])[:, None]

        forward_results =  visit_rep[:-1] * mask
        forward_cross_entropy = -(t[1:] * T.log(forward_results + logEps) + (1. - t[1:]) * T.log(1. - forward_results + logEps))
        backward_results =  results[1:] * mask1
        backward_cross_entropy = -(t[:-1] * T.log(backward_results + logEps) + (1. - t[:-1]) * T.log(1. - backward_results + logEps))

        visit_cost = (forward_cross_entropy.sum(axis=1).sum(axis=0) + backward_cross_entropy.sum(axis=1).sum(axis=0)) / (mask1.sum() + logEps)
        return None

    @tf.function
    def computeConceptCost(self, i_vec, j_vec): 
        logEps = tf.constant(1e-5)
        norms = tf.reduce_sum(tf.math.exp(tf.matmul(self.embedding, self.embedding, transpose_b=True)), axis=1)
        denoms = tf.math.exp(tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(self.embedding, i_vec), 
                                                       tf.nn.embedding_lookup(self.embedding, j_vec)), axis=1))
        concept_cost = tf.negative(tf.math.log((tf.divide(denoms, tf.gather(norms, i_vec)) + logEps)))
        return tf.math.reduce_mean(concept_cost)

    def computeTotalCost(self, x_batch, i_vec, j_vec, label_batch):
        batch_cost = tf.math.add(self.computeVisitCost(x_batch, label_batch), self.computeConceptCost(i_vec, j_vec))
        return batch_cost

    def saveResults(self, save_dir, epoch, avg_loss):
        np.save(os.path.join(save_dir, "phedvec_e{:03d}_loss{:.4f}.npy".format(epoch, avg_loss)),
                np.array(self.embedding[:]))
        saveDictionary(self.concept2id, save_dir, "concept2id.pkl")
        save_loss(self.epoch_loss_avg, save_dir)
        print("Embedding results have been saved")

    def train(self, num_epochs, batch_size, buffer_size, save_dir):
        cost_avg = tf.keras.metrics.Mean()
        dataset = tf.data.Dataset.from_tensor_slices((self.training_data, self.labels)).shuffle(buffer_size).batch(batch_size)
        for epoch in range(num_epochs):
            print("shuffle data and prepare batch for epoch...")
            total_batch = int(np.ceil(len(self.training_data)) / batch_size)
            progbar = tf.keras.utils.Progbar(total_batch)

            for one_batch in dataset:
                i_vec = []
                j_vec = []
                x_batch, label_batch = one_batch
                for x in np.array(x_batch):
                    pickij(x, i_vec, j_vec)
                with tf.GradientTape() as tape:
                    batch_cost = self.computeTotalCost(x_batch, i_vec, j_vec, label_batch)
                gradients = tape.gradient(batch_cost, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                cost_avg(batch_cost)
                progbar.add(1)

            if (epoch % 1) == 0: 
                avg_loss = cost_avg.result()
                print("Epoch {}: Loss: {:.4f}".format(epoch+1, avg_loss))
                self.epoch_loss_avg.append(avg_loss.numpy)
                
        self.saveResults(save_dir, epoch, avg_loss)

def readPatientRecord(file_dir):
    with open(file_dir, "rb") as f:
        mylist = pickle.load(f)
    
    patient_record = []

    print("read patient data...")
    for i in tqdm(range(len(mylist))):
        patient_record.append(mylist[i][0])
    return patient_record

def getUniqueSet(patient_record):
    """--i: patient record
    --o: list of unique concepts in the record"""
    print("get unique concept set...")
    unique_concept_set = set()
    
    for record in patient_record:
        for concept in record:
            unique_concept_set.add(concept)
            
    return unique_concept_set

def buildDict(concept_list):
    print("build concept dict...")
    my_dict = dict()
    for i in range(len(concept_list)):
        my_dict.update({concept_list[i] : i + 1})
    
    return my_dict

def processPatientRecord(patient_record, concept2id):
    print("process training data...")
    print("convert concept to concept ID")
    converted_record = convertToID(patient_record, concept2id)
    print("pad patient record")
    padded_record = padRecord(converted_record)
    return padded_record

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

    return x, mask, i_vec, j_vec