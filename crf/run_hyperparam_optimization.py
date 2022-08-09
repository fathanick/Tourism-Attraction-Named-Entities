from ner_crf import *
from helper.dataset_reader import read_tsv

if __name__ == "__main__":
    data = read_tsv('../dataset/all_data.tsv')
    ner = NamedEntityRecognition()
    ner.hyperparameter_optimization(data)