import numpy as np 
import json
from dotmap import DotMap
import tensorflow as tf

class DataProcessor(object):
    """
    class for preprocessing raw patient records
    """

    def __init__(self, config_dir):
        self.config = set_config(config_dir)

    def process_patientrecord(self, patient_data):
        unique_concepts = getUniqueSet(patient_data)
        self.concept2id = buildDict(list(unique_concepts))
        self.training_patient_record = map_patientrecord(patient_data, self.concept2id)

    def save_data(self):
        pass


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

def map_patientrecord(patient_record, concept2id):
    print("process training data...")
    print("convert concept to concept ID")
    converted_record = convertToID(patient_record, concept2id)
    print("pad patient record")
    padded_record = pad_record(converted_record)
    return padded_record

def convertToID(patient_record, conncept2id):
    converted_record = []
    
    for record in patient_record:
        converted_concepts = []
        for concept in record:
            converted_concepts.append(conncept2id[concept])
        converted_record.append(converted_concepts)
    return converted_record

def pad_record(patient_record, padding_option="post"):
    padded_record = tf.keras.preprocessing.sequence.pad_sequences(patient_record, padding=padding_option)
    return padded_record