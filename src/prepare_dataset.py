# from prepare_data import data_path
from pathlib import Path
import os


def get_split(dataset_path):
    dataset_path = Path(dataset_path)

    train_file_names = []
    val_file_names = []
    test_file_names = []
    sets = os.listdir(dataset_path)
    for set in sets:
        if set == "train":
            train_file_names = list((dataset_path / (set) / 'images').glob('*'))
        elif set == "test":
            test_file_names = list((dataset_path / (set) / 'images').glob('*'))
        else:
            val_file_names = list((dataset_path / (set) / 'images').glob('*'))

    return train_file_names, test_file_names, val_file_names
