import numpy as np
from dotmap import DotMap
import json
import pickle

class EvaluateMCR(object):
    """A class for evaluating different MCRs"""

    def __init__(self, json_dir):
        self.config = set_config(json_dir)
        self.conceptset_dict = dict()

        self.concept2id = None

        self.glove_emb = None
        self.skipgram_emb = None
        self.med2vec_emb = None
        self.phedvec_emb = None

    def setConceptdict(self):
        """intersection concept dict between emb and phenotypes"""
        conceptset_dict_raw = load_dictionary(self.config.data.conceptset_dict)
        concept2id_glove = load_dictionary(self.config.results.concept2id_glove)
        concept2id_skipgram = load_dictionary(self.config.results.concept2id_skipgram)
        concept2id_med2vec = load_dictionary(self.config.results.concept2id_med2vec)
        concept2id_phedvec = load_dictionary(self.config.results.concept2id_phedvec)

        unique_concept_conceptset = count_unique(conceptset_dict_raw)
        unique_concept_glove = set(concept2id_glove.keys())
        unique_concept_skipgram = set(concept2id_skipgram.keys())
        unique_concept_med2vec = set(concept2id_med2vec.keys())
        unique_concept_phedvec = set(concept2id_phedvec.keys())

        glove_emb_raw = np.load(self.config.results.glove_emb)
        skipgram_emb_raw = np.load(self.config.results.skipgram_emb)
        phedvec_emb_raw = np.load(self.config.results.phedvec_emb)
        med2vec_emb_raw = np.load(self.config.results.med2vec_emb)

        intersection_concept = set.intersection(unique_concept_conceptset, unique_concept_glove, unique_concept_skipgram,
        unique_concept_med2vec, unique_concept_phedvec)

        for concept_set in list(conceptset_dict_raw.keys()):
            intersection = set.intersection(set(conceptset_dict_raw[concept_set]), intersection_concept)
            if len(intersection) > 0:
                self.conceptset_dict[concept_set] = list(intersection)

        self.concept2id = build_dict(list(intersection_concept))
        self.glove_emb = rebuild_intersection_emb(self.concept2id, concept2id_glove, glove_emb_raw)
        self.skipgram_emb = rebuild_intersection_emb(self.concept2id, concept2id_skipgram, skipgram_emb_raw)
        self.med2vec_emb = rebuild_intersection_emb(self.concept2id, concept2id_med2vec, med2vec_emb_raw)
        self.phedvec_emb = rebuild_intersection_emb(self.concept2id, concept2id_phedvec, phedvec_emb_raw)

        
    def buildSimilarityMatrix(self):
        pass

    def computeFscore(self, k, mode):
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

def count_unique(my_dict):
    concept_list = []
    for key in list(my_dict.keys()):
        concept_list.extend(my_dict[key])
    
    return list(set(concept_list))

def load_dictionary(data_dir):
    with open(data_dir, 'rb') as f:
        my_dict = pickle.load(f)
    return my_dict

def build_dict(my_list):
    return 

def rebuild_intersection_emb(intersection_dict, concept2id, emb_matrix):
    return