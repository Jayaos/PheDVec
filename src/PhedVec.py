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
        self.config = setConfig(config_dir)
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01)
        self.concept2id = None
        self.training_data = None
        self.labels = None
        self.epoch_loss_avg = [] # record avg loss for all epochs
        
        self.visit_activation = tf.keras.layers.Activation(activation=tf.keras.activations.tanh)
        self.phecode_classifier = tf.keras.layers.Dense(self.config.hparams.num_pheclass, 
        name="phe_classifier", activation=tf.keras.activations.softmax) 
        
    def fitToData(self):
        with tf.device("/cpu:0"):
            patient_data, labels = readPatientRecord(self.config.data.patient_record)
            unique_concepts = getUniqueSet(patient_data)
            self.concept2id = buildDict(list(unique_concepts))
            self.training_data = processPatientRecord(patient_data, self.concept2id)
            self.labels = padLabels(labels)
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

def save_loss(lossfile, save_dir):
    with open(os.path.join(save_dir, "training_loss.txt"), "w") as f:
        for i in range(len(lossfile)):
            f.write("epoch {} : {}\n".format(i+1, lossfile[i]))

def shuffleData(data1, data2):
    shuffle_index = list(range(data1.shape[0]))
    random.shuffle(shuffle_index)
    
    data1_shuffled = tf.gather(data1, shuffle_index)
    data2_shuffled = tf.gather(data2, shuffle_index)
        
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

def padLabels(labels):
    multi_hot = tf.reduce_sum(tf.one_hot(tf.ragged.constant(labels), depth=582), axis=1)
    return multi_hot

def saveDictionary(mydict, save_dir, dict_name):
    with open(os.path.join(save_dir, dict_name), 'wb') as f:
        pickle.dump(mydict, f)