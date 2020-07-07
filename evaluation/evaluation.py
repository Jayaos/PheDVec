import numpy as np
from dotmap import DotMap
import json
import pickle
from collections import OrderedDict
from sklearn.metrics import pairwise_distances

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

        self.glove_simmat = None
        self.skipgram_simmat = None
        self.med2vec_simmat = None
        self.phedvec_simmat = None

        self.glove_precision = OrderedDict()
        self.skipgram_precision = OrderedDict()
        self.med2vec_precision = OrderedDict()
        self.phedvec_precision = OrderedDict()

        self.glove_recall = OrderedDict()
        self.skipgram_recall = OrderedDict()
        self.med2vec_recall = OrderedDict()
        self.phedvec_recall = OrderedDict()

        self.glove_F1score = OrderedDict()
        self.skipgram_F1score = OrderedDict()
        self.med2vec_F1score = OrderedDict()
        self.phedvec_F1score = OrderedDict()

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
        self.glove_simmat = 1 - pairwise_distances(self.glove_emb, metric="cosine")
        self.skipgram_simmat = 1 - pairwise_distances(self.skipgram_emb, metric="cosine")
        self.med2vec_simmat = 1 - pairwise_distances(self.med2vec_emb, metric="cosine")
        self.phedvec_simmat = 1 - pairwise_distances(self.phedvec_emb, metric="cosine")
        
    def computeF1score(self):
        unique_conceptset = list(self.conceptset_dict.keys())

        print("Compute F1-score for GloVe")
        for conceptset in unique_conceptset:
            precision = self.glove_precision[conceptset]
            recall = self.glove_recall[conceptset]
            F1score = 2 * ((precision * recall) / (precision + recall))
            self.glove_F1score[conceptset] = F1score

    def computePrecision(self, k, mode):
        unique_conceptset = list(self.conceptset_dict.keys())

        print("Compute precision for GloVe...")
        for conceptset in unique_conceptset:
            candidate_concepts = self.conceptset_dict[conceptset]
            avg_precision = compute_precision(candidate_concepts, k, self.glove_simmat, 
                                      self.concept2id, mode)
            self.glove_precision.update({conceptset : avg_precision})

    def computeRecall(self, k, mode):
        unique_conceptset = list(self.conceptset_dict.keys())

        print("Compute recall for GloVe...")
        for conceptset in unique_conceptset:
            candidate_concepts = self.conceptset_dict[conceptset]
            avg_recall = compute_recall(candidate_concepts, k, self.glove_simmat, 
                                      self.concept2id, mode)
            self.glove_recall.update({conceptset : avg_recall})

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
    my_dict = dict()

    for i in range(len(my_list)):
        my_dict[my_list[i]] = i

    return my_dict

def rebuild_intersection_emb(intersection_dict, concept2id, emb_matrix):
    emb_dim = emb_matrix.shape[1]
    intersection_emb_matrix = np.zeros((len(intersection_dict), emb_dim))

    for concept in list(intersection_dict.keys()):
        intersection_emb_matrix[intersection_dict[concept]] = emb_matrix[concept2id[concept]]
    
    return intersection_emb_matrix

def compute_precision(candidate_list, k, simmat, simmat_dict, mode):
    candidate_index = set()
    candidate_num = k
    for concept in candidate_list:
        candidate_index.add(simmat_dict[concept])
    
    if mode == "percent":
        candidate_num = int(np.ceil(len(candidate_list) * (k / 100)))
    
    precision_list = []
    for concept in candidate_list:
        retrieved_concepts = set(np.argsort(simmat[simmat_dict[concept]])[(-(candidate_num)-1):-1])
        relevants = len(candidate_index.intersection(retrieved_concepts))
        precision_list.append(relevants)
    
    avg_precision = np.average(np.array(precision_list) / candidate_num)
    
    return avg_precision

def compute_recall(candidate_list, k, simmat, simmat_dict, mode):
    candidate_index = set()
    candidate_num = k
    for concept in candidate_list:
        candidate_index.add(simmat_dict[concept])
    
    if mode == "percent":
        candidate_num = int(np.ceil(len(candidate_list) * (k / 100)))
    
    recall_list = []
    for concept in candidate_list:
        retrieved_concepts = set(np.argsort(simmat[simmat_dict[concept]])[(-(candidate_num)-1):-1])
        relevants = len(candidate_index.intersection(retrieved_concepts))
        recall_list.append(relevants)
    
    avg_recall = np.average(np.array(recall_list) / len(candidate_list))
    
    return avg_recall