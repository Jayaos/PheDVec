import tensorflow as tf
import pickle
from tqdm import tqdm
import json
from dotmap import DotMap

def create_dataset(buffer_size, batch_size, config_dir):
    """Creates a tf.data Dataset.
    Args:
        buffer_size: Shuffle buffer size.
        batch_size: Batch size
        data_dir: data directory
    Returns:
        dataset for training
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    config = set_config(config_dir)

    with tf.device("/cpu:0"):
        patient_record, labels = read_data(config.data.patient_record)
        labels_padded = pad_labels(labels)
        unique_concepts = get_uniqueset(patient_record)
        concept2id = build_dict(list(unique_concepts))
        training_record = process_patientrecord(patient_record, concept2id)
        save_dictionary(concept2id, config.data.concept2id)

    train_dataset = tf.data.Dataset.from_tensor_slices((training_record, labels_padded))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, len(patient_record)

def read_data(file_dir):
    with open(file_dir, "rb") as f:
        mylist = pickle.load(f)
    
    patient_record = []
    labels = []

    print("read patient data...")
    for i in tqdm(range(len(mylist))):
        patient_record.append(mylist[i][0])
        labels.append(mylist[i][1])
    return patient_record, labels

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

def pad_labels(labels):
    multi_hot = tf.reduce_sum(tf.one_hot(tf.ragged.constant(labels), depth=582), axis=1)
    return multi_hot

def get_uniqueset(patient_record):
    """--i: patient record
    --o: list of unique concepts in the record"""
    print("get unique concept set...")
    unique_concept_set = set()
    
    for record in patient_record:
        for concept in record:
            unique_concept_set.add(concept)
            
    return unique_concept_set

def build_dict(concept_list):
    print("build concept dict...")
    my_dict = dict()
    for i in range(len(concept_list)):
        my_dict.update({concept_list[i] : i + 1})
    
    return my_dict

def process_patientrecord(patient_record, concept2id):
    print("process training data...")
    print("convert concept to concept ID")
    converted_record = convert_ID(patient_record, concept2id)
    print("pad patient record")
    padded_record = pad_record(converted_record)
    return padded_record

def convert_ID(patient_record, conncept2id):
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

def load_dictionary(dictionary_dir):
    f = open(dictionary_dir, "rb")
    dict_load = pickle.load(f)
    return dict_load

def save_dictionary(my_dict, save_dir):
    with open(save_dir, "wb") as f:
        pickle.dump(my_dict, f)
    print("concept2id successfully saved in the configured dir")