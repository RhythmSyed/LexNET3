import sys
import numpy as np

import tensorflow as tf
tf.compat.v1.set_random_seed(0)

sys.path.append('../common/')

from lstm_common import (
    vectorize_path,
    load_dataset,
    get_paths,
)
from docopt import docopt
from itertools import count
from evaluation_common import (
    evaluate,
)
from collections import defaultdict
from common.knowledge_resource import KnowledgeResource
from paths_lstm_classifier_tf import PathLSTMClassifier
from train_integrated_tf import load_paths_and_word_vectors


def main():
    args = docopt("""The LSTM-based integrated pattern-based and distributional method for multiclass
    semantic relations classification: Model Test

    Usage:
        test_integrated_tf.py <corpus_prefix> <dataset_prefix> <model_prefix_file>

        <corpus_prefix> = file path and prefix of the corpus files
        <dataset_prefix> = file path of the dataset files containing train.tsv, test.tsv, val.tsv and relations.txt
        <model_prefix_file> = output directory and prefix for the model files
    """)

    corpus_prefix = args['<corpus_prefix>']
    dataset_prefix = args['<dataset_prefix>']
    model_prefix_file = args['<model_prefix_file>']

    np.random.seed(133)

    # Load the relations
    with open(dataset_prefix + '/relations.txt', 'r', encoding='utf-8') as f_in:
        relations = [line.strip() for line in f_in]
        relation_index = {relation: i for i, relation in enumerate(relations)}

    # Load the datasets
    print('Loading the dataset...')
    test_set = load_dataset(dataset_prefix + '/test.tsv', relations)
    y_test = [relation_index[label] for label in list(test_set.values())]
    print('Done!')

    # Load the resource (processed corpus)
    print('Loading the corpus...')
    corpus = KnowledgeResource(corpus_prefix)
    print('Done!')

    # Load Model
    classifier, word_index, pos_index, dep_index, dir_index = PathLSTMClassifier.load_model(model_prefix_file)

    # Load the paths and create the feature vectors
    print('Loading path files...')
    x_y_vectors_test, x_test, _, _, _, _, _, _ = load_paths_and_word_vectors(corpus, list(test_set.keys()), word_index)

    print('Evaluation:')
    pred = classifier.predict(x_test, x_y_vectors=x_y_vectors_test)
    precision, recall, f1, support = evaluate(y_test, pred, relations, do_full_reoprt=True)
    print('Precision: %.3f, Recall: %.3f, F1: %.3f' % (precision, recall, f1))


if __name__ == '__main__':
    main()
