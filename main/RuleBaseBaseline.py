#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:11:13 2023

@author: andrewriznyk
"""
import re
import nltk
from nltk.corpus import wordnet
import pandas as pd
import csv
nltk.download('wordnet')

list_words=['i', "i'm", "i am", "i'd", "i would", "my", 'your', "you"]
list_syn={}
for word in list_words:
    synonyms=[]
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            lem_name = re.sub('[^a-zA-Z0-9 \n\.]', ' ', lem.name())
            #print(lem_name)
            synonyms.append(lem_name)
    list_syn[word]=set(synonyms)
#print (list_syn)


# Building dictionary of Intents & Keywords
keywords={}
keywords_dict={}

keywords['offer']=[]
keywords['ask']=[]

for synonym in list(list_syn['i']):
    keywords['offer'].append('.*\\b'+synonym+'\\b.*')
for synonym in list(list_syn["i'm"]):
    keywords['offer'].append('.*\\b'+synonym+'\\b.*')
for synonym in list(list_syn["i am"]):
    keywords['offer'].append('.*\\b'+synonym+'\\b.*')
for synonym in list(list_syn["i'd"]):
    keywords['offer'].append('.*\\b'+synonym+'\\b.*')
for synonym in list(list_syn["i would"]):
    keywords['offer'].append('.*\\b'+synonym+'\\b.*')
for synonym in list(list_syn["i would"]):
    keywords['offer'].append('.*\\b'+synonym+'\\b.*')
for synonym in list(list_syn["my"]):
    keywords['offer'].append('.*\\b'+synonym+'\\b.*')
    
keywords['offer'].append('.*\\b'+"i'm"+'\\b.*')
keywords['offer'].append('.*\\b'+"i am"+'\\b.*')
keywords['offer'].append('.*\\b'+"i'd"+'\\b.*')
keywords['offer'].append('.*\\b'+"i would"+'\\b.*')    
keywords['offer'].append('.*\\b'+"my"+'\\b.*')
keywords['offer'].append('.*\\b'+"offer"+'\\b.*')

for synonym in list(list_syn['your']):
    keywords['ask'].append('.*\\b'+synonym+'\\b.*')
for synonym in list(list_syn["you"]):
    keywords['ask'].append('.*\\b'+synonym+'\\b.*')
    
keywords['ask'].append('.*\\b'+"your"+'\\b.*')
keywords['ask'].append('.*\\b'+"you"+'\\b.*')

for intent, keys in keywords.items():
    keywords_dict[intent]=re.compile('|'.join(keys))


user_input = "I'm looking to diversify my assets and would like to trade 35 iron for 4 gold. Would you consider it?".lower()
user_input2 = "I'm looking to acquire 35 wood and am willing to offer 33 wheat in exchange. Would you be interested?".lower()

exchangeSyn = ['for', 'central', 'change', 'commutation', 'commute', 'convert', 'exchange', 'interchange', 'rally', 'replace', 'substitute', 'substitution', 'switch', 'switch over', 'telephone exchange']
offer = ["i", "i'm", "i am", "i'd", "i would", "my", "offer"]
ask = ['your', "you"]

commodities = ["gold", "stone", "wood", "wheat", "coal", "iron", "aluminum", "horses"]


def exchangeSplit(sentence):
    allSplits = []
    for exchangeWord in exchangeSyn:
        allSplits.append(sentence.split(exchangeWord))
    return allSplits

def get_distance(w1, w2, sentence):
    words = sentence.split()
    if w1 in words and w2 in words:
        return abs(words.index(w2) - words.index(w1))
    return 0

def extractDigits(sentence):
    items = []
    wordList = sentence.split()
    for x in wordList:
        if x.isdigit():
            items.append(x)
    return items

    num = [int(x) for x in sentence.split() if x.isdigit()] 
    return num

def extractCommodities(sentence):
    items = []
    wordList = sentence.split()
    for x in wordList:
        for comod in commodities:
            if comod in x:
                items.append(x)
    return items

# connect values with commodities via proxmitiy

def connectValuesWithNumbers(sentence):
    itemQuantityList = []
    numbers = extractDigits(sentence)
    commodItems = extractCommodities(sentence)
    for num in numbers:
        minDist = 1000
        minItem = ""
        for item in commodItems:
            dist = get_distance(str(num), item, sentence)
            if dist < minDist and dist != 0:
                minDist = dist
                minItem = item
        itemQuantityList.append((minItem , str(num)))
    return itemQuantityList

def connectValueItemsWithAgent(itemQuantityList, sentence):
    userItemQuantityDict = {}
    userItemQuantityList = []
    for item, quanity in itemQuantityList:
        temp = {}
        offerDist = listDistItr(item, quanity, offer, sentence)
        askDist = listDistItr(item, quanity, ask, sentence)
        temp['offer'] = (offerDist, item, quanity)
        temp['ask'] = (askDist, item, quanity)
        userItemQuantityList.append(temp)
    #print(userItemQuantityList)
    #min offer
    minOffer = 1000
    for agentData in userItemQuantityList:
        if agentData['offer'][0] < minOffer:
            minOffer = agentData['offer'][0]
            userItemQuantityDict['offer'] = (agentData['offer'][1], agentData['offer'][2])
            
    minAsk = 1000
    for agentData in userItemQuantityList:
        if agentData['ask'][0] < minAsk:
            minAsk = agentData['ask'][0]
            userItemQuantityDict['ask'] = (agentData['ask'][1], agentData['ask'][2])
    return userItemQuantityDict

def listDistItr(item, quanity, agentList, sentence):
    minDist = 1000
    for word in agentList:
        distQuaant = get_distance(str(quanity), word, sentence)
        distItem = get_distance(item, word, sentence)
        dist = distItem if distItem < distQuaant else distQuaant
        if dist < minDist and dist != 0:
            minDist = dist
    return minDist

def cleanCommodities(userItemQuantityDict):
    for agent in userItemQuantityDict:
        for commod in commodities:
            if commod in userItemQuantityDict[agent][0]:
                userItemQuantityDict[agent] = (commod, userItemQuantityDict[agent][1]) 
    return userItemQuantityDict

offerCountCorrect = 0
offerCountIncorrect = 0
offerItemCorrect = 0
offerItemIncorrect = 0

askCountCorrect = 0
askCountIncorrect = 0
askItemCorrect = 0
askItemIncorrect = 0

offerNotFound = 0
askNotFound = 0

gold = []
pred = []

with open('../data/trades.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for i, line in enumerate(reader):
        itemQuany = connectValuesWithNumbers(line[0].lower())
        agentPred = connectValueItemsWithAgent(itemQuany, line[0].lower())
        prediction = cleanCommodities(agentPred)
        
        if 'offer' in prediction:
            if line[1] == prediction['offer'][1]:
                offerCountCorrect += 1
                
                pred.append(1)
                gold.append(1)
            else:
                offerCountIncorrect += 1
                pred.append(1)
                gold.append(0)
            if line[2] == prediction['offer'][0]:
                offerItemCorrect += 1
                pred.append(2)
                gold.append(2)
            else:
                offerItemIncorrect += 1
                pred.append(2)
                gold.append(0)

        else:
            offerNotFound += 1
            
            pred.append(0)
            gold.append(1)
            pred.append(0)
            gold.append(2)
            
        if 'ask' in prediction:
            if line[3] == prediction['ask'][1]:
                askCountCorrect += 1
                pred.append(3)
                gold.append(3)
            else:
                askCountIncorrect += 1
                pred.append(3)
                gold.append(0)
            if line[4] == prediction['ask'][0]:
                askItemCorrect += 1
                pred.append(4)
                gold.append(4)
            else:
                pred.append(4)
                gold.append(0)
                askItemIncorrect += 1
        else:
            askNotFound +=1

            pred.append(0)
            gold.append(3)
            pred.append(0)
            gold.append(4)
    

print("")
print("Accuracy of Offer Value")
print(offerCountCorrect / (offerCountCorrect + offerCountIncorrect))
print("Accuracy of Offer Item Type")
print(offerItemCorrect / (offerItemCorrect + offerItemIncorrect))
print("")
print("")
print("Accuracy of Asking Value")
print(askCountCorrect / (askCountCorrect + askCountIncorrect))
print("Accuracy of Asking Item Type")
print(askItemCorrect / (askItemCorrect + askItemIncorrect))           
print("")
print("")
# not sure what to call the 2ed agent that the offer is being made to
print("Percentage Not Found / Detected. Total for Asking OR offer")
print((offerNotFound + askNotFound) / (askCountCorrect + askCountIncorrect + askItemCorrect + askItemIncorrect))


from sklearn.metrics import classification_report
print(classification_report(gold, pred))