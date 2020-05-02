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
        self.config = setConfig(config_dir) # apply json load function later
        self.optimizer = tf.keras.optimizers.Adadelta() # set hparams later
        self.concept2id = None
        self.training_data = None
        
        self.embedding_layer = None
        self.visit_activation = tf.keras.layers.Activation(activation=tf.keras.activations.tanh)
        self.phecode_classifier = tf.keras.layers.Dense(582, name="phe_classifier", activation=tf.keras.activations.softmax) 
        # output dim is the number of phecode classes, which is 582
        
    def fitToData(self):
        print("load patient record...")
        patient_data = open_patient_record(self.config.data.patient_record)
        print("get unique concept set...")
        unique_concepts = getUniqueSet(patient_data)
        print("build concept dict...")
        self.concept2id = buildDict(list(unique_concepts))
        print("process training data...")
        self.training_data = processPatientRecord(patient_data, self.concept2id)
        
    def initModel(self):
        self.embedding_layer = tf.keras.layers.Embedding(len(self.concept2id)+1, 1024, mask_zero=True)
    
    def get_visitRep(self, x):
        emb_output = self.embedding_layer(x)
        visit_rep = self.visit_activation(tf.reduce_sum(tf.ragged.boolean_mask(emb_output, emb_output._keras_mask), axis=1))
        return visit_rep
        
    def computeVisitCost(self, x_batch, label_batch):
        # visit-level cost
        visit_rep = self.get_visitRep(x_batch)
        prediction = self.phecode_classifier(visit_rep)
        mhot_labels = tf.reduce_sum(tf.one_hot(tf.ragged.constant(label_batch), depth=582), axis=1)
        logEps = tf.constant(1e-5)
        cost1 = tf.multiply(mhot_labels, tf.math.log(tf.math.add(prediction, logEps)))
        cost2 = tf.multiply(tf.math.subtract(1, mhot_labels), tf.math.log(tf.math.add(tf.math.subtract(1,prediction), logEps)))
        batch_cost = tf.math.divide( tf.math.negative(tf.reduce_sum(tf.math.add(cost1, cost2))), len(x_batch))
        return batch_cost
    
    def computeConceptCost(self, x_batch):# has not yet tested
        w_emb = self.embedding_layer(range(len(self.concept2id)))
        norms = tf.math.exp(tf.reduce_sum(tf.matmul(w_emb, w_emb, transpose_b=True), axis=1))
        denoms = tf.math.exp(tf.reduce_sum(tf.multiply(self.embedding_layer(i_vec), self.embedding_layer(j_vec)), axis=1))
        concept_cost = tf.negative(tf.math.log((tf.divide(denoms, norms[i_vec]) + logEps)))
        return concept_cost
    
    def compute_gradients(self, x_batch, label_batch):
        with tf.GradientTape() as tape:
            cost = self.compute_cost(x_batch, label_batch)
                
        return cost, tape.gradient(cost, self.trainable_variables)
        
    def train_model(self, num_epochs, batch_size, save_dir):
        train_data = read_list(self.config.data.train_data) 
        
        for epoch in range(num_epochs):
            print("shuffle data and prepare batch for epoch")
            shuffled_data = shuffle_data(train_data)
            x_raw, labels = splitTrainData(shuffled_data)
            x = convert_concept_batch(x_raw, self.concept2id)
            total_batch = int(np.ceil(len(x) / batch_size))
            
            progbar = tf.keras.utils.Progbar(total_batch)
            for i in range(total_batch):
                x_batch = x[i * batch_size : (i+1) * batch_size]
                label_batch = labels[i * batch_size : (i+1) * batch_size]
                cost, gradients = self.compute_gradients(x_batch, label_batch)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                progbar.add(1)

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

def load_dictionary(pklfile):
    f = open(pklfile, "rb")
    dict_load = pickle.load(f)
    return dict_load

def load_emb_matrix(npydir):
    return np.load(npydir)

def shuffle_data(data):
    train_data_shuffled = []
    shuffle_index = list(range(len(data)))
    random.shuffle(shuffle_index)
    
    for ind in shuffle_index:
        train_data_shuffled.append(data[ind])
        
    return train_data_shuffled

def splitTrainData(mydata):
    x = []
    y = []
    
    for i in range(len(mydata)):
        x.append(mydata[i][0])
        y.append(mydata[i][1])
    
    return x, y

def convert_concept(record, concept2id):
    converted_record = []
    for concept in record:
        converted_record.append(concept2id[concept])
        
    return converted_record

def convert_concept_batch(record_batch, concept2id):
    converted_batch = []
    for record in record_batch:
        converted_batch.append(convert_concept(record, concept2id))
        
    return tf.keras.preprocessing.sequence.pad_sequences(converted_batch, padding="post")

def readPatientRecord(saved_dir):
    with open(saved_dir, "rb") as f:
        mylist = pickle.load(f)
    return mylist

def pick_ij(visit_record):
    i_vec = []
    j_vec = []
    for first in visit_record:
        for second in visit_record:
            if first == second: continue
            i_vec.append(first)
            j_vec.append(second)
            
    return i_vec, j_vec

def readPatientRecord(saved_dir):
    with open(saved_dir, "rb") as f:
        mylist = pickle.load(f)
    return mylist

def getUniqueSet(patient_record):
    """--i: patient record
    --o: list of unique concepts in the record"""
    unique_concept_set = set()
    
    for record in patient_record:
        for concept in record:
            unique_concept_set.add(concept)
            
    return unique_concept_set

def buildDict(concept_list):
    my_dict = dict()
    for i in range(len(concept_list)):
        my_dict.update({concept_list[i] : i + 1})
    
    return my_dict
    
def processPatientRecord(patient_record, concept2id):
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