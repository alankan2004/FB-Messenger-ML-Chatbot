import nltk
## To stem the words
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    hi
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_words = []
    docs_tags = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # stemming takes each word and bring it down to the root word.
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_words.append(wrds)
            docs_tags.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_words):
        bag = []

        wrds = [stemmer.stem(w) for w in doc if w != "?"]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_tags[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.reset_default_graph()

## each training input are the same length
## the network starts at the input data then to hidden layer
neural_network = tflearn.input_data(shape=[None, len(training[0])])

## 8 neurons at hidden layer
neural_network = tflearn.fully_connected(neural_network, 8)
neural_network = tflearn.fully_connected(neural_network, 8)

## softmax gives a probabilities for each output
## output is [0/1, 0/1]
neural_network = tflearn.fully_connected(neural_network, len(output[0]), activation="softmax")
neural_network = tflearn.regression(neural_network)

model = tflearn.DNN(neural_network)

try:
    ffff
    model.load("model.tflearn")
except:
    ## number of epoch is number of times it will see the same data
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    for word_in_sentence in sentence_words:
        for idx, word in enumerate(words):
            if word == word_in_sentence:
                ## 1 rep the word exists
                bag[idx] = 1

    return np.array(bag)

def chat():
    print("Start talking with the bot (type quit to leave)!")
    while  True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        ## [0] since there's  only one result in the result list
        results = model.predict([bag_of_words(inp, words)])[0]
        print(results)
        ## index of the highest prob in the array
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("Sorry I didn't get that, try again.")

if __name__ == "__main__":
    chat()
