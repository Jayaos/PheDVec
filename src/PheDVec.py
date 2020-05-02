import tensorflow as tf
import json
from dotmap import DotMap
import numpy as np
import pickle
import random
import gensim
from tqdm import tqdm

class PheDVec(tf.keras.Model):
    def __init__(self, config_dir):
        super(PheDVec, self).__init__()
        self.config = setConfig(config_dir)
        self.optimizer = tf.keras.optimizers.Adadelta()
        self.concept2id = None
        self.training_data = None
        self.labels = None
        self.epoch_loss_avg = [] # record avg loss for all epochs
        
        self.embedding_layer = None
        self.visit_activation = tf.keras.layers.Activation(activation=tf.keras.activations.tanh)
        self.phecode_classifier = tf.keras.layers.Dense(582, name="phe_classifier", activation=tf.keras.activations.softmax) 
        # output dim is the number of phecode classes, 582
        
    def fitToData(self):
        patient_data, labels = readPatientRecord(self.config.data.patient_record)
        unique_concepts = getUniqueSet(patient_data)
        self.concept2id = buildDict(list(unique_concepts))
        self.training_data = processPatientRecord(patient_data, self.concept2id)
        self.labels = labels
        print("fitting process has been completed")
        
    def initModel(self):
        if self.concept2id == None: 
            print("set concept2id before initialzing the model")

        print("initialize model...")
        self.embedding_layer = tf.keras.layers.Embedding(len(self.concept2id)+1, 1024, mask_zero=True)
    
    def getVisitRep(self, x_batch):
        emb_output = self.embedding_layer(x_batch) # n(batch_size) * l(padded_len)
        visit_rep = self.visit_activation(tf.reduce_sum(tf.ragged.boolean_mask(emb_output, emb_output._keras_mask), axis=1))
        return visit_rep # n(batch_size) * d(dim)
        
    def computeVisitCost(self, x_batch, label_batch):
        visit_rep = self.getvisitRep(x_batch)
        phecode_prediction = self.phecode_classifier(visit_rep)
        mhot_labels = tf.reduce_sum(tf.one_hot(tf.ragged.constant(label_batch), depth=582), axis=1)
        logEps = tf.constant(1e-5)
        visit_cost1 = tf.multiply(mhot_labels, tf.math.log(tf.math.add(phecode_prediction, logEps)))
        visit_cost2 = tf.multiply(tf.math.subtract(1, mhot_labels), tf.math.log(tf.math.add(tf.math.subtract(1,phecode_prediction), logEps)))
        visit_cost = tf.math.divide( tf.math.negative(tf.reduce_sum(tf.math.add(visit_cost1, visit_cost2))), len(x_batch))
        return visit_cost
    
    def computeConceptCost(self, i_vec, j_vec): # has not yet tested
        w_emb = self.embedding_layer(range(len(self.concept2id)))
        logEps = tf.constant(1e-5)
        norms = tf.math.exp(tf.reduce_sum(tf.matmul(w_emb, w_emb, transpose_b=True), axis=1))
        denoms = tf.math.exp(tf.reduce_sum(tf.multiply(self.embedding_layer(i_vec), self.embedding_layer(j_vec)), axis=1))
        concept_cost = tf.negative(tf.math.log((tf.divide(denoms, norms[i_vec]) + logEps)))
        return concept_cost

    def computeCost(self, x_batch, i_vec, j_vec, label_batch):
        batch_cost = tf.math.reduce_sum(self.computeVisitCost(x_batch, label_batch), self.computeConceptCost(i_vec, j_vec))
        return batch_cost
    
    def computeGradients(self, x_batch, i_vec, j_vec, label_batch):
        with tf.GradientTape() as tape:
            cost = self.computeCost(x_batch, i_vec, j_vec, label_batch)

        return cost, tape.gradient(cost, self.trainable_variables)
        
    def train(self, num_epochs, batch_size, save_dir):
        cost_avg = tf.keras.metrics.Mean()
        for epoch in range(num_epochs):
            print("shuffle data and prepare batch for epoch...")
            x_training, labels_training = shuffleData(self.training_data, self.labels)
            total_batch = int(np.ceil(len(x_training)) / batch_size)
            progbar = tf.keras.utils.Progbar(total_batch)

            for i in range(total_batch):
                i_vec = []
                j_vec = []
                x_batch = x_training[i * batch_size : (i+1) * batch_size]
                label_batch = labels_training[i * batch_size : (i+1) * batch_size]
                for x in x_batch:
                    pickij(x, i_vec, j_vec)
                batch_cost, gradients = self.computeGradients(x_batch, i_vec, j_vec, label_batch)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                cost_avg(batch_cost) 
                progbar.add(1)
                print("Step {}: Loss: {:.4f}".format(self.optimizer.iterations.numpy(), batch_cost))

            if (epoch % 1) == 0: 
                avg_loss = cost_avg.result()
                print("Epoch {}: Loss: {:.4f}".format(epoch+1, avg_loss))
                self.epoch_loss_avg.append(avg_loss)

# Functions 
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

def shuffleData(data1, data2):
    data1_shuffled = []
    data2_shuffled = []
    shuffle_index = list(range(len(data1)))
    random.shuffle(shuffle_index)
    
    for ind in shuffle_index:
        data1_shuffled.append(data1[ind])
        data2_shuffled.append(data2[ind])
        
    return data1_shuffled, data2_shuffled

def readPatientRecord(file_dir):
    with open(file_dir, "rb") as f:
        mylist = pickle.load(f)
    
    patient_record = []
    labels = []

    print("read patient data...")
    for i in tqdm(range(len(mylist))):
        patient_record.append(mylist[i][0])
        labels.append(mylist[i][1])

    return patient_record, labels

def pickij(visit_record, i_vec, j_vec):
    unpadded_record = visit_record[visit_record != 0]
    for first in unpadded_record:
        for second in unpadded_record:
            if first == second: continue
            i_vec.append(first)
            j_vec.append(second)

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

def convertToID(patient_record, conncept2id):
    converted_record = []
    
    for record in patient_record:
        converted_concepts = []
        for concept in record:
            converted_concepts.append(conncept2id[concept])
        converted_record.append(converted_concepts)
    return converted_record

def padRecord(patient_record, padding_option="post"):
    padded_record = tf.keras.preprocessing.sequence.pad_sequences(patient_record, padding=padding_option)
    return padded_record

def open_patient_record(data_dir):
    patient_record = []
    with open(data_dir, "r") as f:
        patients = f.read().split("\n")
        for i in tqdm(range(len(patients))):
            patient = patients[i]
            patient_record.append(patient.split(","))
    return patient_record