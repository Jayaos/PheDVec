import numpy as np
from dotmap import DotMap
import json
import pickle

class EvaluateMCR(object):
    """A class for evaluating different MCRs"""

    def __init__(self, json_dir):
        self.config = set_config(json_dir)

        self.concept2id_glove = load_dictionary(self.config.results.concept2id_glove)
        self.concept2id_skipgram = load_dictionary(self.config.results.concept2id_skipgram)
        self.concept2id_med2vec = load_dictionary(self.config.results.concept2id_med2vec)
        self.concept2id_phedvec = load_dictionary(self.config.results.concept2id_phedvec)

        self.glove_emb = np.load(self.config.results.glove_emb)
        self.skipgram_emb = np.load(self.config.results.skipgram_emb)
        self.med2vec_emb = np.load(self.config.results.med2vec_emb)
        self.phedvec_emb = np.load(self.config.results.phedvec_emb)

    def setPhedict(self):
        """intersection concept dict between emb and phenotypes"""
        pass

def set_config(json_file):
    """
    Get config data from a json file
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        json_body = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = DotMap(json_body)
    return config      

def load_dictionary(data_dir):
    with open(data_dir, 'rb') as f:
        my_dict = pickle.load(f)
    return my_dict
