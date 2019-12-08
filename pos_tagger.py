import argparse
import time

import nltk
import os
import random
from collections import defaultdict
from nltk.corpus import brown
import joblib
import numpy as np


class POSTagger():
    def __init__(self):
        self.pos_dict = {}
        self.word_dict = {}
        self.pos_to_index = {}
        self.word_to_index = {}
        self.initial = None
        self.transition = None
        self.emission = None
        self.UNK = '*UNKNOWN*'
        # the length of pos:
        self.P = 0
        # the length of word:
        self.W = 0
        self.init_dict()

    def init_dict(self):
        word_set = set()
        pos_set = set()
        word_set.add(self.UNK)

        for word, pos in brown.tagged_words():
            word_set.add(word)
            pos_set.add(pos)

        word_list = list(word_set)
        pos_list = list(pos_set)

        for i, word in enumerate(word_list):
            self.word_dict[i] = word
            self.word_to_index[word] = i

        for i, pos in enumerate(pos_list):
            self.pos_dict[i] = pos
            self.pos_to_index[pos] = i

        self.W = len(self.word_dict)
        self.P = len(self.pos_dict)

    def train(self, train_set):
        initial_count = np.zeros((1, self.P))
        transition_count = np.zeros((self.P, self.P))
        emission_count = np.zeros((self.P, self.W))

        pre_pos_index = 0

        for sent in train_set:
            for i, word_and_tag in enumerate(sent):
                word, pos = word_and_tag
                word_index = self.word_to_index[word]
                pos_index = self.pos_to_index[pos]
                emission_count[pos_index, word_index] += 1
                if i == 0:
                    initial_count[0, pos_index] += 1
                else:
                    transition_count[pre_pos_index, pos_index] += 1
                pre_pos_index = pos_index

        self.initial = np.log((initial_count + 1) / np.sum(initial_count + 1, axis=1))
        self.transition = np.log((transition_count + 1) / np.sum(transition_count + 1, axis=1))
        self.emission = np.log((emission_count + 1) / np.sum(emission_count + 1, axis=1).reshape((self.P, 1)))

    def viterbi(self, sentence):
        T = len(sentence)
        v = np.zeros((self.P, T))
        backpointer = np.zeros((self.P, T))

        # initialization step
        v[:, 0, None] = self.initial.transpose() + self.emission[:, sentence[0], None]
        backpointer[:, 0] = 0
        # recursion step
        for t in range(1, T):
            v_t_candidate = (v[:, t - 1, None] + self.transition).transpose() + self.emission[:, sentence[t], None]
            v[:, t, None] = np.max(v_t_candidate, axis=1).reshape((self.P, 1))
            backpointer[:, t, None] = np.argmax(v_t_candidate, axis=1).reshape((self.P, 1))
        # termination step
        pos_index = np.argmax(v[:, T - 1, None])
        index_path = [0] * T
        for t in range(T - 1, -1, -1):
            index_path[t] = pos_index
            pos_index = backpointer[int(pos_index), t]
        best_path = []
        for index in index_path:
            best_path.append(self.pos_dict[index])
        return best_path

    def predict(self, tokens):
        sentence = []
        for word in tokens:
            if word not in self.word_to_index:
                word = self.UNK
            sentence.append(self.word_to_index[word])
        return list(zip(tokens, self.viterbi(sentence)))

    def evaluate(self, test_set):
        total = 0
        correct = 0
        for sent in test_set:
            tokens, gold_poses = zip(*sent)
            _, valid_poses = zip(*self.predict(tokens))
            for gold_pos, valid_pos in zip(gold_poses, valid_poses):
                if gold_pos == valid_pos:
                    correct += 1
                total += 1
        return correct / total


def train_POSTagger():
    # split data into train and test
    all_data = list(brown.tagged_sents())
    random.shuffle(all_data)
    split_index = int(0.98 * len(all_data))
    train_set = all_data[:split_index]
    test_set = all_data[split_index:]

    output_path = "./pos_tagger.jbl"
    start_time = time.time()
    pos_tagger = POSTagger()
    pos_tagger.train(train_set)
    joblib.dump(pos_tagger, output_path)
    end_time = time.time()
    print(f'Creating Bayes classifier in {output_path}')

    print(f'\tElapsed time: {end_time - start_time}s')

    print(f'\tAccuracy: {pos_tagger.evaluate(test_set)}')


def run_POSTagger(input_text):
    pos_tagger = joblib.load("pos_tagger.jbl")
    tokens = nltk.word_tokenize(input_text)
    print(f'* input text: {input_text}')
    print(f'* pairs of word and pos: ')
    for token, pos in pos_tagger.predict(tokens):
        print(f'\t{token} -- {pos}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', dest='train')
    group.add_argument('--run', action='store', dest='input_text')

    args = parser.parse_args()

    if args.train:
        train_POSTagger()

    if args.input_text:
        run_POSTagger(args.input_text)

# if __name__ == '__main__':
#     pos = POSTagger()
#
#     # make sure these point to the right directories
#     pos.train()
#     print(pos.predict('I hate you !'.split(' ')))
#     print('Accuracy:', pos.evaluate())
