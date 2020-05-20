import tensorflow as tf
import pickle
from tqdm import tqdm
import json
from dotmap import DotMap

def create_dataset(buffer_size, batch_size, data_format, data_dir):
    """Creates a tf.data Dataset.
    Args:
        buffer_size: Shuffle buffer size.
        batch_size: Batch size
        data_dir: data directory
    Returns:
        dataset for training
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    patient_record, labels = read_data(data_dir)
    train_dataset = tf.data.Dataset.from_tensor_slices((patient_record, labels))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset

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

def load_dictionary(dictionary_dir):
    f = open(dictionary_dir, "rb")
    dict_load = pickle.load(f)
    return dict_load