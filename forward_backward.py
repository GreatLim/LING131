import numpy as np
from pos_tagger import POSTagger


class ForwardBackward(POSTagger):
    def __init__(self):
        self.pos_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        self.pos_to_index = {pos: j for j, pos in self.pos_dict.items()}
        self.word_dict = {}
        self.word_to_index = {}
        self.initial = None
        self.transition = None
        self.emission = None
        self.T = 0
        self.build_data_set()
        # the length of pos:
        self.P = len(self.pos_dict)
        # the length of word:
        self.W = len(self.word_dict)

    def build_data_set(self):
        with open('data_set') as f:
            word_set = set()
            for line in f:
                words = line.split()
                self.T = len(words)
                for word in words:
                    word_set.add(word)
            word_list = list(word_set)

            for i, word in enumerate(word_list):
                self.word_dict[i] = word
                self.word_to_index[word] = i

    def forward(self, sentence):
        T = len(sentence)
        alpha = np.zeros((self.P, T))
        # alpha initialization step
        alpha[:, 0, None] = self.initial.reshape((self.P, 1)) * self.emission[:, sentence[0], None]
        # alpha recursion step
        for t in range(1, T):
            for i in range(self.P):
                for j in range(self.P):
                    alpha[j, t] += alpha[i, t - 1] * self.transition[i, j] * self.emission[j, sentence[t]]

        # no alpha termination step necessary
        return alpha

    def backward(self, sentence):
        T = len(sentence)
        beta = np.zeros((self.P, T))
        # beta initialization step
        beta[:, T - 1, None] = 1
        # beta recursion step
        for t in range(T - 2, -1, -1):
            for i in range(self.P):
                for j in range(self.P):
                    beta[i, t] += self.transition[i, j] * self.emission[j, sentence[t + 1]] * beta[j, t + 1]

        # no beta termination step necessary

        return beta

    def generate_random_matrix(self, rows, cols):
        matrix = np.zeros((rows, cols))

        for i in range(rows):
            randint = np.random.randint(10000, size=cols)
            row = randint / np.sum(randint)
            matrix[i, :] = row

        return matrix

    def forward_backward(self, data_set):
        self.initial = self.generate_random_matrix(1, self.P)
        self.transition = self.generate_random_matrix(self.P, self.P)
        self.emission = self.generate_random_matrix(self.P, self.W)

        max_step = 30
        step = 0
        sentences_index = []
        converged = False
        with open(data_set) as f:
            sentences = f.read().splitlines()
            for sentence in sentences:
                sentence_index = []
                for word in sentence.split():
                    sentence_index.append(self.word_to_index[word])
                sentences_index.append(sentence_index)

        # iterate until convergence
        while not converged:
            trainsition_numerator = np.zeros((self.P, self.P))
            trainsition_denominator = np.zeros((self.P, self.P))
            emission_numerator = np.zeros((self.P, self.W))
            emission_denominator = np.zeros((self.P, self.W))

            # E-step
            for sentence_index in sentences_index:

                gamma_numerator = np.zeros((self.P, self.T))
                xi_numerator = np.zeros((self.P, self.P, self.T - 1))

                alpha = self.forward(sentence_index)
                beta = self.backward(sentence_index)

                denominator = np.sum(alpha[:, self.T - 1])

                gamma_numerator += alpha * beta

                for i in range(self.P):
                    for j in range(self.P):
                        for t in range(self.T - 1):
                            xi_numerator[i, j, t] = alpha[i, t] * self.transition[i, j] * self.emission[
                                j, sentence_index[t + 1]] * beta[j, t + 1]

                # xi_numerator = 1
                gamma = gamma_numerator / denominator
                xi = xi_numerator / denominator

                # M-step
                self.initial += gamma[:, 0].reshape((1, self.P))

                for i in range(self.P):
                    for t in range(self.T - 1):
                        for j in range(self.P):
                            trainsition_numerator[i, j] += xi[i, j, t]
                            trainsition_denominator[i, :] += xi[i, j, t]

                for j in range(self.P):
                    for w in range(self.W):
                        for t in range(self.T):
                            if w == sentence_index[t]:
                                emission_numerator[j, w] += gamma[j, t]
                            emission_denominator[j, w] += gamma[j, t]

            self.transition = trainsition_numerator / trainsition_denominator
            self.emission = emission_numerator / emission_denominator
            self.initial = self.initial / len(sentences_index)

            step += 1
            if step > max_step:
                converged = True

        self.transfer_to_log()

    def transfer_to_log(self):
        smooth = np.nextafter(0, 1)
        self.initial = np.log(self.initial + smooth)
        self.emission = np.log(self.emission + smooth)
        self.transition = np.log(self.transition + smooth)

    def viterbi(self, sentence):
        T = len(sentence)
        v = np.full((self.P, T), -np.inf)
        backpointer = np.full((self.P, T), -np.inf)

        # initialization step
        v[:, 0] = self.initial + self.emission[:, sentence[0]]
        backpointer[:, 0] = 0
        # recursion step
        for t in range(1, T):
            v_t_candidate = (v[:, t - 1, None] + self.transition).transpose() + self.emission[:, sentence[t], None]
            v[:, t, None] = np.max(v_t_candidate, axis=1).reshape((self.P, 1))
            backpointer[:, t, None] = np.argmax(v_t_candidate, axis=1).reshape((self.P, 1))
            # print(v)
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

    def run(self, data_set):
        with open(data_set) as f:
            sentences = f.read().splitlines()
            print('* Dataset: ')
            for sentence in sentences:
                print(f'\t{sentence}')
            print('* POS dict:', self.pos_dict)
            print('* Word dict:', self.word_dict)
            print('* Initial probability distribution:', np.exp(self.initial))
            print('* Transition matrix:', np.exp(self.transition))
            print('* Emission matrix:', np.exp(self.emission))
            reverse_word_dict = {word: j for j, word in self.word_dict.items()}
            print('* Result:')
            for sentence in sentences:
                if sentence:
                    word_list = sentence.split()
                    index_list = [reverse_word_dict[word] for word in word_list]
                    pos_list = self.viterbi(index_list)
                    print('\t' + ' '.join([word + '/' + pos
                                           for word, pos in zip(word_list, pos_list)]))


if __name__ == '__main__':
    fb = ForwardBackward()
    fb.forward_backward('data_set')
    fb.run('data_set')
