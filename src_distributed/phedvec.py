import tensorflow as tf
import numpy as np
import pickle
import random
import gensim
from tqdm import tqdm

from src.utils import set_config, load_dictionary

class PhedVec(tf.keras.Model):
    """PhedVec model that embeds medical concepts and gets visit embedding"""
    def __init__(self, config_dir):
        super(PhedVec, self).__init__()
        self.config = set_config(config_dir)
        self.concept2id = load_dictionary(self.config.data.concept2id)
        self.label_num = self.config.params.label_num
        self.embedding_dim = self.config.params.embedding_dim

        self.embedding = tf.Variable(tf.random.uniform([len(self.concept2id)+1, self.embedding_dim], 0.1, -0.1))
        self.visit_activation = tf.keras.layers.Activation(activation=tf.keras.activations.tanh)
        self.phecode_classifier = tf.keras.layers.Dense(self.label_num, activation=tf.keras.activations.softmax) 

    def call(self, x):
        emb_output = tf.nn.embedding_lookup(self.embedding, x) # n(batch_size) * l(padded_len)
        mask = x != 0
        visit_rep = self.visit_activation(tf.reduce_sum(tf.ragged.boolean_mask(emb_output, mask), axis=1))
        softmax_var = self.phecode_classifier(visit_rep)
        return softmax_var # n(batch_size) * label_num