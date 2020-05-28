import numpy as np 
import json
from dotmap import DotMap
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

class Preprocess(object):
    """
    class for preprocessing raw patient records
    """

    def __init__(self, config_dir):
        self.config = set_config(config_dir)
        
    def process_rawrecord(self, removing_list=None):
        """
        process raw condition_occurrence.rpt into source visit list and standard concept visit list
        --i: raw condition_occurrence.rpt file
        --o: source concept visit list file and standard concept visit list file
        """

        record_df = build_dataframe(self.config.raw_data.condition_occurrence)
        if removing_list != None:
            record_df = remove_concepts(removing_list, record_df)
        self.source_visitlist, self.standrad_visitlist = process_visitlist(record_df)

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

def build_dataframe(data_dir):
    """
    convert raw condition_occurrence data to dataframe
    """
    print("reading data...")
    with open(data_dir, "r") as f:
        body = f.read()
        body = body.split("\n")
        concepts = body[2:-4]
    
    print("processing data...")
    concept_record = []
    for i in tqdm(range(len(concepts))):
        raw_pair = concepts[i].split(" ")
        while ("" in raw_pair):
            raw_pair.remove("")
        if raw_pair[1] != "0":
            concept_record.append(raw_pair)
    
    record_df = pd.DataFrame(concept_record)
    record_df = record_df[[0,1,2,3]]
    record_df = record_df.rename(columns={0 : "patient_id", 1 : "concept_id", 2 : "visit_date", 3 : "source_id"})
    return record_df

def remove_concepts(removing_list, filtering_df):
    """
    removing provided concepts from the dataframe
    """
    for concept in removing_list:
        filtering_df = filtering_df[filtering_df["concept_id"] != concept]
        
    return filtering_df

def process_visitlist(record_df):
    """
    process dataframe into source concept visit list and standard concept visit list
    """
    print("grouping patients by patient_id and slicing by visit window")
    grouped_df = record_df.groupby(["patient_id", "visit_date"])
    groups = list(grouped_df.groups.keys())
    
    source_visit_list = []
    standard_visit_list = []
    
    for i in tqdm(range(len(groups))):
        if len(set(record_df.loc[grouped_df.groups[groups[i]]]["source_id"])) > 1 and len(set(record_df.loc[grouped_df.groups[groups[i]]]["concept_id"])) > 1:
            source_visit_list.append(list(set(record_df.loc[grouped_df.groups[groups[i]]]["source_id"])))
            standard_visit_list.append(list(set(record_df.loc[grouped_df.groups[groups[i]]]["concept_id"])))
    
    return source_visit_list, standard_visit_list