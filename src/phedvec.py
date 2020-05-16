import tensorflow as tf
import json
from dotmap import DotMap
import numpy as np
import pickle
import random
import gensim
from tqdm import tqdm

class PhedVec(tf.keras.Model):
    """PhedVec model that embeds medical concepts and gets visit embedding"""
    def __init__(self, config_dir):
        super(PhedVec, self).__init__()
        self.concept2id = None
        self.label_num = None # read from config
        self.embedding_dim = None # read from config

        self.embeddings = tf.Variable(tf.random.uniform([len(self.concept2id)+1, 1000], 0.1, -0.1))
        self.visit_activation = tf.keras.layers.Activation(activation=tf.keras.activations.tanh)
        self.phecode_classifier = tf.keras.layers.Dense(self.label_num, activation=tf.keras.activations.softmax) 

    def call(self, x, training=True):
        emb_output = tf.nn.embedding_lookup(self.embedding, x) # n(batch_size) * l(padded_len)
        mask = x != 0
        visit_rep = self.visit_activation(tf.reduce_sum(tf.ragged.boolean_mask(emb_output, mask), axis=1))
        return visit_rep # n(batch_size) * d(dim)