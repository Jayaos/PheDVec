import tensorflow as tf
import pickle
from tqdm import tqdm

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