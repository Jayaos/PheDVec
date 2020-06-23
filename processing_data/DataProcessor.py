import numpy as np 
import pandas as pd
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
        self.icd_omop_dict = None
        self.filtered_standard_record = None
        self.filtered_source_record = None

    def processRawRecord(self, removing_list=None):

        if self.icd_omop_dict == None:
            print("build icd_omop_dict first")

        record_df = build_record_df(self.config.data.patient_record)

        if removing_list != None:
            print("remove concepts in the list from the patient record")
            record_df = remove_concepts(removing_list, record_df)
        else:
            print("No concepts to remove from the patient record")

        standard_record, source_record = process_record(record_df)
        self.filtered_standard_record, self.filtered_source_record = filter_record(standard_record, 
        source_record, self.icd_omop_dict)

    def buildDict_ICDPhecode(self):

        print("load ICD-phecode mapping data...")
        icd9_phecode_map = pd.read_csv(self.config.data.icd9_phecode_map, encoding="ISO-8859-1")
        icd10_phecode_map = pd.read_csv(self.config.data.icd10_phecode_map, encoding="ISO-8859-1")
        icd10cm_phecode_map = pd.read_csv(self.config.data.icd10cm_phecode_map, encoding="ISO-8859-1")

        print("build ICD-phecode dictionary...")
        self.icd9_phecode_dict = build_ICD9phecode_dict(icd9_phecode_map)
        self.icd10_phecode_dict = build_ICD10phecode_dict(icd10_phecode_map)
        self.icd10cm_phecode_dict = build_ICD10cmphecode_dict(icd10cm_phecode_map)

    def buildDict_ICDOMOP(self):

        print("load ICD-omop mapping data...")
        icd9_omop = read_icd_omop(self.config.data.icd9_omop)
        icd10_omop = read_icd_omop(self.config.data.icd10_omop)
        icd10cm_omop = read_icd_omop(self.config.data.icd10cm_omop)

        icd_omop = pd.concat([icd9cm_omop, icd10_omop, icd10cm_omop], ignore_index=True)
        
        print("build ICD-phecode dictionary...")
        self.icd_omop_dict = build_ICDOMOP_dict(icd_omop)


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

def build_ICD9phecode_dict(icd9_map):
    icd9_dict = dict()
    for i in tqdm(range(icd9_map.shape[0])):
        icd9_dict.update({icd9_map["icd9"][i] : icd9_map["phecode"][i]})
        
    return icd9_dict

def build_ICD10phecode_dict(icd10_map):
    icd10_dict = dict()
    for i in tqdm(range(icd10_map.shape[0])):
        icd10_dict.update({icd10_map["ICD10"][i] : icd10_map["PHECODE"][i]})
    
    return icd10_dict

def build_ICD10cmphecode_dict(icd10cm_map):
    icd10cm_dict = dict()
    for i in tqdm(range(icd10cm_map.shape[0])):
        icd10cm_dict.update({icd10cm_map["icd10cm"][i] : icd10cm_map["phecode"][i]})
    
    return icd10cm_dict

def build_ICDOMOP_dict(omop_raw):
    mydict = dict()
    for i in tqdm(range(omop_raw.shape[0])):
        mydict.update({omop_raw["concept_id"][i] : (omop_raw["source_type"][i], omop_raw["source_id"][i])})
    
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

    grouped_by_patient = record_df.groupby(["patient_id"])
    patient_group = list(grouped_by_patient.groups)

    for patient in patient_group:
        visit_under_patient = record_df.loc[grouped_by_patient.groups[patient]]
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

def filter_record(standard_record, source_record, icd_omop_dict):
    assert len(standard_record) == len(source_record), "the length of the two records must be the same"
    available_source_concepts = set(icd_omop_dict.keys())
    filtered_standard_record = []
    filtered_source_record = []

    for i in tqdm(range(len(source_record))):
        filtered_standard_visit = []
        filtered_source_visit = []

        for k in range(len(source_record[i])):
            if len(set.intersection(set(source_record[i][k]), available_source_concepts)) > 1:
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