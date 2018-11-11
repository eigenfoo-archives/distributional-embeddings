import sys
import pickle
import re


def process_text(text_file, pickle_name, THRESH):
    frequency_dict = {}
    word_dict = {}
    count = 0
    with open(text_file, "r") as text:
        for line in text:
            words = re.sub(r"\.+", " ", line).split()
            # print(words)
            for word in words:
                if word in word_dict:
                    frequency_dict[word] += 1
                else:
                    word_dict[word] = count
                    count += 1
                    frequency_dict[word] = 1
    word_dict = {word: word_dict[word]
                 for word in word_dict if frequency_dict[word] > THRESH}
    pickle.dump(word_dict, open(pickle_name, "wb"))


if __name__ == "__main__":
    # Usage, pass in the text file, then the name of the pickle file for the
    # dictionary, and then the thresh for word frequency
    text_file = sys.argv[1]
    pickle_name = sys.argv[2]
    THRESH = int(sys.argv[3])
    print("Starting to process")
    process_text(text_file, pickle_name, THRESH)
    print("Done Processing")
