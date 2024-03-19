import nltk
import numpy as np

with open('../../data/tradeAndNonTradePrompts.csv', newline='') as csvfile:
    lines = (line for line in csvfile)
    fullFile = np.loadtxt(lines, delimiter=',', skiprows=1, usecols=(0), dtype=object)

tokens = nltk.word_tokenize(fullFile[0])
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
print(entities)