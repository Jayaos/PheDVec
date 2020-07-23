import numpy as np 
import pandas as pd
import os
import pickle
import json
from dotmap import DotMap
import tensorflow as tf
from tqdm import tqdm

class DataProcessor(object):
    """
    class for preprocessing raw patient records
    """

    def __init__(self, config_dir):
        self.config = set_config(config_dir)
        self.icd9_phecode_dict = None
        self.icd10_phecode_dict = None
        self.icd10cm_phecode_dict = None
        self.omop_icd_dict = None

        self.standard_record = None
        self.source_record = None

        self.concept2id = None
        self.phecode2id = None
        self.med2vec_format = None
        self.phedvec_format = None

    def processRawRecord(self, removing_list=None):

        if self.omop_icd_dict == None:
            print("build omop_icd_dict first")

        record_df = build_record_df(self.config.data.patient_record)

        if removing_list != None:
            print("remove concepts in the list from the patient record")
            record_df = remove_concepts(removing_list, record_df)
        else:
            print("No concepts to remove from the patient record")

        print("processing record...")
        standard_record, source_record = process_record(record_df) # this is the bottleneck process
        self.standard_record, self.source_record = filter_record(standard_record, 
        source_record, self.omop_icd_dict) 

    def convertRecord(self, padding=False):
        print("generate phecode label based on omop_source_id...")
        label_source = label_record(self.source_record, self.omop_icd_dict, 
        self.icd10_phecode_dict, self.icd10cm_phecode_dict)

        self.phecode2id = build_dict(label_source)
        self.concept2id = build_dict(self.standard_record)

        self.med2vec_format = convert_med2vec_format(self.standard_record, self.concept2id, padding=padding)
        self.phedvec_format = convert_phedvec_format(self.standard_record, label_source, self.concept2id, self.phecode2id, padding=padding)

    def buildDict_ICDPhecode(self, decimal):

        print("load ICD-phecode mapping data...")
        icd10_phecode_map = pd.read_csv(self.config.data.icd10_phecode_map, encoding="ISO-8859-1")
        icd10cm_phecode_map = pd.read_csv(self.config.data.icd10cm_phecode_map, encoding="ISO-8859-1")

        print("build ICD-phecode dictionary...")
        self.icd10_phecode_dict = build_ICD10phecode_dict(icd10_phecode_map, decimal)
        self.icd10cm_phecode_dict = build_ICD10cmphecode_dict(icd10cm_phecode_map, decimal)

    def buildDict_OMOPICD(self):

        print("load ICD-omop mapping data...")
        icd10_omop = read_icd_omop(self.config.data.icd10_omop)
        icd10cm_omop = read_icd_omop(self.config.data.icd10cm_omop)

        icd_omop = pd.concat([icd10_omop, icd10cm_omop], ignore_index=True)
        
        print("build ICD-phecode dictionary...")
        icd_phecode_set = set.union(set(self.icd10_phecode_dict.keys()), set(self.icd10cm_phecode_dict.keys()))
        self.omop_icd_dict = build_OMOPICD_dict(icd_omop, icd_phecode_set) 

    def saveResults(self):
        print("save concept2id and phecode2id in the specified dir")
        save_data(self.concept2id, "concept2id.pkl", self.config.dir.save_dir)
        save_data(self.phecode2id, "phecode2id.pkl", self.config.dir.save_dir)

        print("save training data in the specified dir")
        save_data(self.med2vec_format, "med2vec_training.pkl",self.config.dir.save_dir)
        save_data(self.phedvec_format, "phedvec_training.pkl",self.config.dir.save_dir)

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

def get_unique_concept(record):
    unique_concept = set()

    print("count unique concept in the record...")
    for i in tqdm(range(len(record))):
        for visit in record[i]:
            for concept in visit:
                unique_concept.add(concept)

    return unique_concept

def build_dict(record):
    concept_list = list(get_unique_concept(record))

    print("build concept dict...")
    my_dict = dict()
    for i in range(len(concept_list)):
        my_dict.update({concept_list[i] : i})

    return my_dict

def read_icd_omop(data_dir):
    print("reading data")
    with open(data_dir, "r") as f:
        body = f.read()
        body = body.split("\n")
        concepts = body[2:-4]
    
    print("processing data")
    concept_record = []
    for i in tqdm(range(len(concepts))):
        raw_pair = concepts[i].split(" ")
        while ("" in raw_pair):
            raw_pair.remove("")
        if raw_pair[0] != "0":
            concept_record.append(raw_pair)
    
    record_df = pd.DataFrame(concept_record)
    record_df = record_df[[0,1,2]]
    record_df = record_df.rename(columns={0 : "concept_id", 1 : "source_type", 2 : "source_id"})
    
    return record_df

def build_ICD9phecode_dict(icd9_map, decimal):
    icd9_dict = dict()
    for i in tqdm(range(icd9_map.shape[0])):
        if np.isnan(icd9_map["phecode"][i]) != True:
            icd9_dict.update({icd9_map["icd9"][i] : truncate_decimal(icd9_map["phecode"][i], decimal=decimal)})
        
    return icd9_dict

def build_ICD10phecode_dict(icd10_map, decimal):
    icd10_dict = dict()
    for i in tqdm(range(icd10_map.shape[0])):
        if np.isnan(icd10_map["PHECODE"][i]) != True:
            icd10_dict.update({icd10_map["ICD10"][i] : truncate_decimal(icd10_map["PHECODE"][i], decimal=decimal)})
    
    return icd10_dict

def build_ICD10cmphecode_dict(icd10cm_map, decimal):
    icd10cm_dict = dict()
    for i in tqdm(range(icd10cm_map.shape[0])):
        if np.isnan(icd10cm_map["phecode"][i]) != True:
            icd10cm_dict.update({icd10cm_map["icd10cm"][i] : truncate_decimal(icd10cm_map["phecode"][i], decimal=decimal)})
    
    return icd10cm_dict

def build_OMOPICD_dict(omop_raw, icd_phecode_set):
    mydict = dict()
    for i in tqdm(range(omop_raw.shape[0])):
        if omop_raw["source_id"][i] in icd_phecode_set:
            mydict.update({omop_raw["concept_id"][i] : (omop_raw["source_type"][i], omop_raw["source_id"][i])})
        else:
            continue
    
    return mydict

def build_record_df(data_dir):
    print("reading data")
    with open(data_dir, "r") as f:
        body = f.read()
        body = body.split("\n")
        concepts = body[2:-4]
    
    print("processing data")
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
    
    for concept in removing_list:
        filtered_df = filtering_df[filtering_df["concept_id"] != concept]
        
    return filtered_df

def process_record(record_df):
    standard_record = []
    source_record = []

    print("grouping patient record by patient_id...")
    grouped_by_patient = record_df.groupby(["patient_id"])
    patient_group = list(grouped_by_patient.groups)

    for i in tqdm(range(len(patient_group))):
        visit_under_patient = record_df.loc[grouped_by_patient.groups[patient_group[i]]]
        grouped_by_visit = visit_under_patient.groupby(["visit_date"])
        visit_group = list(grouped_by_visit.groups)
        
        standard_patient = []
        source_patient = []

        for visit in visit_group:
            standard_concepts = set(visit_under_patient.loc[grouped_by_visit.groups[visit]]["concept_id"])
            source_concepts = set(visit_under_patient.loc[grouped_by_visit.groups[visit]]["source_id"])
            if len(standard_concepts) > 1 and len(source_concepts) > 1:
                standard_patient.append(list(standard_concepts))
                source_patient.append(list(source_concepts))
            else:
                continue

        standard_record.append(standard_patient)
        source_record.append(source_patient)

    return standard_record, source_record

def filter_record(standard_record, source_record, omop_icd_dict):
    assert len(standard_record) == len(source_record), "the length of the two records must be the same"
    available_source_concepts = set(omop_icd_dict.keys())
    filtered_standard_record = []
    filtered_source_record = []

    for i in tqdm(range(len(source_record))):
        filtered_standard_visit = []
        filtered_source_visit = []

        for k in range(len(source_record[i])):
            if len(set.intersection(set(source_record[i][k]), available_source_concepts)) == len(set(source_record[i][k])):
                filtered_standard_visit.append(standard_record[i][k])
                filtered_source_visit.append(source_record[i][k])
            else:
                continue

        if len(filtered_source_visit) > 1:
            filtered_standard_record.append(filtered_standard_visit)
            filtered_source_record.append(filtered_source_visit)
        else:
            continue

    return filtered_standard_record, filtered_source_record

def label_record(source_record, omop_icd_dict, icd10_phecode_dict, icd10cm_phecode_dict):
    label_record = []
    for i in tqdm(range(len(source_record))):
        label_patient = []
        for k in range(len(source_record[i])):
            label_visit = []
            for source_id in source_record[i][k]:
                dict_type, icd_code = lookup_omop_icd(source_id, omop_icd_dict)
                phecode = lookup_icd_phecode(dict_type, icd_code, icd10_phecode_dict, icd10cm_phecode_dict)
                label_visit.append(phecode)
            label_visit = list(set(label_visit))
            if len(label_visit) > 0:
                label_patient.append(label_visit)
        label_record.append(label_patient)
    
    return label_record

def lookup_omop_icd(source_id, omop_icd_dict):
    return omop_icd_dict[source_id] # return dict_type, icd_code

def lookup_icd_phecode(dict_type, icd_code, icd10_phecode_dict, icd10cm_phecode_dict):
    if dict_type == "ICD10":
        try:
            phecode = icd10_phecode_dict[icd_code]
        except:
            phecode = icd10cm_phecode_dict[icd_code]
    elif dict_type == "ICD10CM":
        try:
            phecode = icd10cm_phecode_dict[icd_code]
        except:
            phecode = icd10_phecode_dict[icd_code]
    
    return phecode

def convert_med2vec_format(standard_record, concept2id, padding=False):
    med2vec_record = []

    print("convert patient record into Med2Vec pacakage compatiable format")
    for i in tqdm(range(len(standard_record))):
        for standard_visit in standard_record[i]:
            coded_visit = apply_concept2id(standard_visit, concept2id)
            med2vec_record.append(coded_visit)
        if i != (len(standard_record) - 1):
            med2vec_record.append([-1])
    
    if padding:
        med2vec_record = tf.keras.preprocessing.sequence.pad_sequences(med2vec_record, padding="post")

    return med2vec_record

def convert_phedvec_format(standard_record, label, concept2id, phecode2id, padding=False):
    assert len(standard_record) == len(label), "Length of the standard record and label must be the same"
    phedvec_record = []
    phedvec_label = []

    for i in tqdm(range(len(standard_record))):
        for standard_visit in standard_record[i]:
            coded_visit = apply_concept2id(standard_visit, concept2id)
            phedvec_record.append(coded_visit)

    for i in tqdm(range(len(label))):
        for l in label[i]:
            coded_label = apply_concept2id(l, phecode2id)
            phedvec_label.append(coded_label)

    if padding:
        phedvec_record = tf.keras.preprocessing.sequence.pad_sequences(phedvec_record, padding="post")
        phedvec_label = tf.keras.preprocessing.sequence.pad_sequences(phedvec_label, padding="post")

    return [phedvec_record, phedvec_label]

def truncate_decimal(values, decimal=0):
    return np.trunc(values*10**decimal)/(10**decimal)

def apply_concept2id(visit, concept2id):
    converted_visit = []
    for concept in visit:
        converted_visit.append(concept2id[concept])
    
    return converted_visit

def save_data(mydata, name, save_dir):
    with open(os.path.join(save_dir, name), 'wb') as f:
        pickle.dump(mydata, f)