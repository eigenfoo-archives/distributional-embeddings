import pickle
import numpy as np
import re

class Data:
    def __init__(self,window,negative,data_file, dictionary_pickle, shuffled_file):
        self.window = window
        self.negative= negative
        self.data_file = open(data_file,"r")
        self.setence = []
        self.dictionary = pickle.load(open(dictionary_pickle, 'rb'))
        self.non_word = -1
        self.shuffled_file = open(shuffled_file, 'r')
        self.sentence_loc = 0
        hold  = re.sub(r"\.+", ".", self.data_file.read(100000)).strip()
        if hold == '':
            raise Exception("file is finished, no more training data!")
        self.buffer = hold.split(".")
        self.shuffled_buffer = self.shuffled_file.read(100000).split()
        self.end_buffer = len(self.buffer)
        self.sentenence_loc = 0
        self.location = 0
        self.sentence = self.buffer[0].split()
        self.sentence_length = len(self.sentence)
        self.shuff_size = len(self.shuffled_buffer)
    def update_sentence(self):
        if self.sentence_loc == (self.end_buffer-2):
            hold  = re.sub(r"\.+", ".", self.data_file.read(100000)).strip()
            if hold == '':
                raise Exception("file is finished, no more training data!")
            self.buffer = hold.split(".")
            self.shuffled_buffer = self.shuffled_file.read(100000).split()
            self.end_buffer = len(self.buffer)
            self.sentence_loc = 0
            self.location = 0
            self.sentence = self.buffer[0].split()
            self.sentence_length = len(self.sentence)
            self.shuff_size = len(self.shuffled_buffer)
        if self.location == self.sentence_length:
            self.sentence_loc+=1
            self.sentence = self.buffer[self.sentence_loc].split()
            self.sentence_length = len(self.sentence)
            self.location = 0
        if self.sentence_length == 0:
            self.update_sentence()
    def next_sample(self):
        self.update_sentence()
        word = self.sentence[self.location]
        #is this risky?
        while word not in self.dictionary:
            self.location+=1
            self.update_sentence()
            word = self.sentence[self.location]
        start = self.location - self.window
        end =  self.location + self.window + 1
        pad = 0
        window_words =[]
        if (start) < 0 :
            window_words = [self.non_word] * (self.window-self.location)
            start = 0
        if end > self.sentence_length:
            pad = end - self.sentence_length
            end = self.sentence_length

        #print(self.location,start,end, self.window-self.location)
        for n in self.sentence[start:end]:
            if n in self.dictionary:
              if n!= word:
                window_words.append(self.dictionary[n])
            else:
                window_words.append(self.non_word)
        window_words+= [self.non_word] * pad
        center_word = self.dictionary[self.sentence[self.location]]
        self.location+=1
        shuff_choices =[]
        for choice in np.random.choice(self.shuff_size, self.negative,replace=False):
            shuff_choices.append(self.shuffled_buffer[choice])
        negative_words= [self.dictionary[n] if n in self.dictionary else self.non_word for n in shuff_choices]
        return window_words,negative_words, center_word



data = Data(3,5,"/home/jonny/Documents/mlfinal/data/data.txt", "pickle.c", "/home/jonny/Documents/mlfinal/data/shuffled.txt")

for i in range(1647734):
    print(data.next_sample())
