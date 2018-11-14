import pickle
import numpy as np
import re
import sys

class Data:
    def __init__(
            self,
            window,
            negative,
            data_file,
            dictionary_pickle):
        self.window = window
        self.negative = negative
        self.data_file = open(data_file, "r")
        self.setence = []
        self.dictionary = pickle.load(open(dictionary_pickle, 'rb'))
        self.dictionary_length = len(self.dictionary)
        self.non_word = -1
        self.sentence_loc = 0
        hold = re.sub(r"\.+", ".", self.data_file.read(100000)).strip()
        if hold == '':
            raise Exception("file is finished, no more training data!")
        self.buffer = hold.split(".")
        self.end_buffer = len(self.buffer)
        self.sentenence_loc = 0
        self.location = 0
        self.sentence = self.buffer[0].split()
        self.sentence_length = len(self.sentence)

    def update_sentence(self):
        if self.sentence_loc == (self.end_buffer - 2):
            hold = re.sub(r"\.+", ".", self.data_file.read(100000)).strip()
            if hold == '':
                raise Exception("file is finished, no more training data!")
            self.buffer = hold.split(".")
            self.end_buffer = len(self.buffer)
            self.sentence_loc = 0
            self.location = 0
            self.sentence = self.buffer[0].split()
            self.sentence_length = len(self.sentence)
        if self.location == self.sentence_length:
            self.sentence_loc += 1
            self.sentence = self.buffer[self.sentence_loc].split()
            self.sentence_length = len(self.sentence)
            self.location = 0
        if self.sentence_length == 0:
            self.update_sentence()

    def next_sample(self):
        self.update_sentence()
        word = self.sentence[self.location]
        # is this risky?
        while word not in self.dictionary:
            self.location += 1
            self.update_sentence()
            word = self.sentence[self.location]
        start = self.location - self.window
        end = self.location + self.window + 1
        pad = 0
        window_words = []
        if (start) < 0:
            # window_words = [self.non_word] * (self.window - self.location)
            start = 0
        if end > self.sentence_length:
            # pad = end - self.sentence_length
            end = self.sentence_length

        for n in self.sentence[start:end]:
            if n in self.dictionary:
                if n != word:
                    window_words.append(self.dictionary[n])
            else:
               # window_words.append(self.non_word)
               pass
       # window_words += [self.non_word] * pad
        center_word = self.dictionary[self.sentence[self.location]]
        self.location += 1
        negative_indices = np.random.choice(self.dictionary_length, len(window_words),replace=False)
        negative_words = []
        for n in negative_indices:
            if n in window_words:
                while n in window_words:
                    n = np.random.randint(self.dictionary_length)
            negative_words.append(n)
        return window_words, negative_words, center_word




if __name__ == "__main__":
    data_location  = sys.argv[1]
    pickle_location = sys.argv[2]
    number_of_samples = sys.argv[3]
    output_file = sys.argv[4]
    window = int(sys.argv[5])
    data = Data(window,window, data_location,pickle_location)
    out = open(output_file,"w")
    for i in range(int(number_of_samples)):
        out.write("{}\n".format(data.next_sample()))
