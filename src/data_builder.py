import pickle
import numpy as np
import re
import sys


class Data:
    """
     Incrementally reads text from a file,
     then iterates through sentences in that text.

     Parameters
     ----------
        window : int
            size of window to look at the left and right of center word
        data_file: str
            name of file that contains all of the text
        dictionary_pickle: str
            name of pickle file that contains the dictionary that maps
                word to id
    """
    def __init__(
            self,
            window,
            data_file,
            dictionary_pickle):
        self.window = window
        self.data_file = open(data_file, "r")
        self.dictionary = pickle.load(open(dictionary_pickle, 'rb'))
        self.dictionary_length = len(self.dictionary)
        # sentence_loc is the sentence within the buffer we are currently at
        self.sentence_loc = 0
        # here we are removing periods and also ellipsis,
        # following this all we have is text
        hold = re.sub(r"\.+", ".", self.data_file.read(100000)).strip()
        self.buffer = hold.split(".")
        self.end_buffer_loc = len(self.buffer)
        # Location is the place within the sentence we are at
        self.location = 0
        self.sentence_words = self.buffer[0].split()
        self.sentence_length = len(self.sentence_words)

    def _update_sentence(self):
        """
          helper function for next_sample,
             it checks if we are finished with the current sentence.
             If we are it updates the sentence to the next sentence
             in the buffer, if the buffer if finished it draws more
             text.
        """
        if self.sentence_loc == (self.end_buffer_loc - 2):
            hold = re.sub(r"\.+", ".", self.data_file.read(100000)).strip()
            if hold == '':
                raise Exception("file is finished, no more training data!")
            self.buffer = hold.split(".")
            self.end_buffer_loc = len(self.buffer)
            self.sentence_loc = 0
            self.location = 0
            self.sentence_words = self.buffer[0].split()
            self.sentence_length = len(self.sentence_words)
        if self.location == self.sentence_length:
            self.sentence_loc += 1
            self.sentence_words = self.buffer[self.sentence_loc].split()
            self.sentence_length = len(self.sentence_words)
            self.location = 0

    def next_sample(self):
        """
            Responsible for getting the relevant inforamtion from the current
                position in the buffer and updating the pointers


            Returns
            ---------
                window_words: array of 2*window ints
                   an array of the ids corresponding to the words in the
                   context of the center word
                negative_words: array of 2*window ints
                   an array of ids of negativly sampled words that are
                   not in the window of the context word or the center
                    word itself.
                center_word: int
                    the id corresponding to the center word
        """
        self._update_sentence()
        word = self.sentence_words[self.location]
        while word not in self.dictionary:
            self.location += 1
            self._update_sentence()
            word = self.sentence_words[self.location]
        start = self.location - self.window
        end = self.location + self.window + 1
        window_words = []
        if (start) < 0:
            start = 0
        if end > self.sentence_length:
            end = self.sentence_length
        for n in self.sentence_words[start:end]:
            if n in self.dictionary:
                if n != word:
                    window_words.append(self.dictionary[n])
        center_word = self.dictionary[word]
        self.location += 1
        negative_indices = np.random.choice(self.dictionary_length,
                                            len(window_words), replace=False)
        negative_words = []
        for i in negative_indices:
            if i in window_words or i == center_word:
                while i in window_words or i == center_word:
                    i = np.random.randint(self.dictionary_length)
            negative_words.append(i)
        return window_words, negative_words, center_word


if __name__ == "__main__":
    """
     Usage:
       python data_builder.py <LOCATION OF TEXT FILE>
        <LOCATION OF DICTIONARY PICKLE> <NUMBER OF SAMPLES DESIRED>
        <OUTPUT FILE NAME> <WINDOW SIZE>
    """
    data_location = sys.argv[1]
    pickle_location = sys.argv[2]
    number_of_samples = sys.argv[3]
    output_file = sys.argv[4]
    window = int(sys.argv[5])
    data = Data(window, data_location, pickle_location)
    out = open(output_file, "w")
    for i in range(int(number_of_samples)):
        out.write("{}\n".format(data.next_sample()))
