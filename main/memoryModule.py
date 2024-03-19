import numpy as np
from sklearn import preprocessing
import math
import random
import time
from scipy.special import softmax
from symbolVocabulary import SymbolVocabulary
import matplotlib.pyplot as plt
import seaborn as sns

#This is globally scoped (like a static instance)
#Yes, probably a bit dangerous, but it's legible and only expect 1 memory module
#at a time; pre-set hyperparams will not change
HyperParams = {
    "temperature" : math.sqrt(2) * 0.25,
    "decay" : 0.5,
    "decayRate" : 0.3,
    "mismatchPenalty" : 2.0,
    "noiseMean" : 0,
    "noiseVariance" : 0.25,
}

#Default parameter settings - not hyper params so separate
DefaultParams = {
    "embeddingLength" : 8,
    "topk" : 5,
}

#Memory vector symbol format: [player1, stone, 1, aiPlayer, iron, 2, success, 2]
#[offeringPlayer, offeringItem, offeringAmt, receivingPlayer, receivingItem, receivingAmt, success, rewardAmt]

class MemoryModule:
    #hyperParams is struct of equation parameters used by memory system
    def __init__(self, vocab):
        self.vectordb = None
        self.vocab = vocab
        self.embLength = DefaultParams["embeddingLength"]
        self.numMemories = 0
        self.memDict = {}
        vocabSize = vocab.GetVocabSize()
        #TODO- make these just ones
        self.symbolRelations = np.ones((vocabSize, vocabSize))
        #Used to keep track of short term memory chains
        self.shortTermMemory = np.array([])

    #embMemVec is a (1,self.embLength) vector; it is a new memory we are adding to db
    #returns True if successfully added memory, False if failed
    def AddEmbeddedMemory(self, embMemVec):
        if (embMemVec.size != self.embLength):
            return False

        #Handle empty db case
        if self.numMemories == 0:
            self.vectordb = np.zeros((1,self.embLength), dtype=int)
            self.vectordb[0] = embMemVec
            self.numMemories += 1
            self.memDict[0] = Memory(embMemVec, self.vocab)
            return True
        
        #TODO - update this to using GetMatchingMemory func call
        #Check if memory already exists and if it does raise activation value
        dupMemIndices = self.checkForDuplicateMemory(embMemVec)
        if dupMemIndices != None:
            dupMemIdx = dupMemIndices[0]
            self.memDict[dupMemIdx].UseMemory()
            return True

        self.vectordb = np.vstack([self.vectordb, embMemVec])
        self.memDict[self.numMemories] = Memory(embMemVec, self.vocab)
        self.numMemories += 1
        return True
    
    def CreateEmbMemVec(self, symbolStream):
        embVec = np.zeros((1,self.embLength), dtype=int)
        for idx, symbol in enumerate(symbolStream):
            embVec[0][idx] = self.vocab.GetEmbValue(str(symbol))

        return embVec
    
    #Debug tooling to print vector db for inspection
    def printVecDB(self):
        print(self.vectordb)
        return
    
    def AddToShortTermMemory(self, embMemVec):
        if self.numMemories != 0:
            dupeMemoryIndices = self.checkForDuplicateMemory(embMemVec)
            
            if dupeMemoryIndices != None:
                print("attemtping to add a duplicate short term memory")
                return

        if self.shortTermMemory.shape[0] == 0:
            self.shortTermMemory = np.array(embMemVec)
        else:
            self.shortTermMemory = np.vstack([self.shortTermMemory, embMemVec]) 
        
        return

    def ClearShortTermMemory(self):
        if self.shortTermMemory.shape[0] != 0:
            for mem in self.shortTermMemory:
                self.AddEmbeddedMemory(np.reshape(mem, (1,self.embLength)))

        self.shortTermMemory = np.array([])

    def GetItemNamesFromEmbeddings(self, startIdx, endIdx, sortedRelationIndicesByRow):
        symbolRelatednessSortedAscending = np.vectorize(self.convertEmbeddingsToSymbols)(sortedRelationIndicesByRow)
        return symbolRelatednessSortedAscending

    #NOTE - your start index should be relative to first index being 1 (0 is undefined token)
    def GetItemRelationsSortedMemoryGraph(self, startIdx, endIdx):
        #Since my vocabulary defines the 0 token as UNDEFINED, if startIdx specified as 0 override
        if startIdx == 0:
            startIdx = 1
        itemRelationSubgraph = self.symbolRelations[startIdx:endIdx+1,startIdx:endIdx+1]
        sortedRelationIndicesByRow = np.argsort(itemRelationSubgraph)
        #convert column index to its embedding value
        sortedRelationIndicesByRow += startIdx
        return sortedRelationIndicesByRow
    
    #Assumes x and y labels are the same and occur in the same order
    def PlotMemoryGraph(self, fileName, xYLabels, title="Token Relatedness in Memory Module", path="../data/Figures/MemoryModule/"):
        #xAndYLabels = ["iron", "stone", "gold", "aluminum", "horses"]
        
        #plt.imshow(self.symbolRelations[0:5,0:5], cmap='hot', interpolation='nearest')
        fig, ax = plt.subplots()
        ax = sns.heatmap(self.symbolRelations[1:6, 1:6], linewidths=0.5, xticklabels=xYLabels, yticklabels=xYLabels)
        plt.title(title)
        path += fileName
        plt.savefig(path)
        plt.show()
        

    def convertEmbeddingsToSymbols(self, inVal):
        #Add one to symbol conversion here because first symbol is (unknown,0)
        #input is 0 indexed so have to offset conversion by 1 to be accurate
        #maybe not the best explanation but think it through!
        return self.vocab.GetSymbol(inVal)

    #Adjusts the knowledge graph represented here by adj matrix according to new memory vector
    #embMemVec is the new memory we are using to update knowledge graph weights
    def UpdateMemoryGraph(self, embMemVec, inTopKMems=np.array([]), topk=DefaultParams["topk"]):
        if len(self.shortTermMemory) != 0:
            for shortTermMem in self.shortTermMemory:
                if not (shortTermMem==embMemVec).all():
                    self.adjustKnowledgeGraphSimilarity(shortTermMem, embMemVec)

        mostSimilarMemories = inTopKMems
        
        if inTopKMems.shape[0] != 0:
            mostSimilarMemories = self.TopKSimilarVectors(embMemVec, topk)
        

        for mem in mostSimilarMemories:
            if not (mem==embMemVec).all():
                self.adjustKnowledgeGraphSimilarity(mem, embMemVec)

        return

    #TODO - probably can optimize this, figure out some tricks
    #Get top k similar vectors to input vector from db
    #embMemVec is (1,embLength) vector we want to find similar vectors to
    #topk is number of similar vectors we want to look for
    def TopKSimilarVectors(self, embMemVec, topk=DefaultParams["topk"]):
        if self.numMemories == 0:
            return np.array([])

        numStoredMemories = self.vectordb.shape[0]
        if topk > numStoredMemories:
            topk = numStoredMemories

        contextuallyWeightedVecDB = np.zeros((numStoredMemories, self.embLength))
        #print(contextuallyWeightedVecDB.shape)

        #for each vocab type, get weights associated with vocab we're comparing to in stored vecdb
        for idx, weights in enumerate(self.symbolRelations[embMemVec.flatten()]):
            print(weights[self.vectordb[:,idx]])
            contextuallyWeightedVecDB[:,idx] = weights[self.vectordb[:,idx]]

        contextuallyWeightedVecDB = contextuallyWeightedVecDB.T
        similarityScores = np.sum(contextuallyWeightedVecDB, axis=0)

        #Add to similarity scores count of how many of the same tokens exist in memories and 
        #trade to compare; this is in the case that we have not established enough relations
        #between tokens and serves as a preliminary/fallback guide for relevant memories
        for idx, _ in enumerate(similarityScores):
            similarityScores[idx] += len(np.intersect1d(self.vectordb[idx], embMemVec))

        topkIndices = np.argpartition(similarityScores, -topk)[-topk:]
        return self.vectordb[topkIndices]
    
    #Adjust the knowledge graph weighted similarity between an old memory and new memory
    def adjustKnowledgeGraphSimilarity(self, oldMemory, newMemory):
        oldMemory = oldMemory.flatten()
        newMemory = newMemory.flatten()
        
        #How relateded each new memory token is to overall accumulated old memory relatedness
        overallNewTokenRelatednessToOldMem = np.zeros((self.embLength))
        #Averaged sum of all relatedness values of old memory to every token in new memory
        oldMemTokenToNewMemAvgRelatedness = np.zeros((self.embLength))

        for newMemIdx, embVal in enumerate(newMemory):
            for oldMemIdx, oldMemVal in enumerate(oldMemory):
                newMemTokenToOldMemTokenRelatedness = self.symbolRelations[embVal][oldMemVal]
                #average relatedness of each old memory token to every new memory token
                #Gives you average relatedness of each old memory token to each new memory token
                oldMemTokenToNewMemAvgRelatedness[oldMemIdx] += newMemTokenToOldMemTokenRelatedness
                #raw relatedness score of each new memory token to old memory
                #Gives you overall relatedness of new memory token to the entire old memory
                overallNewTokenRelatednessToOldMem[newMemIdx] += newMemTokenToOldMemTokenRelatedness
        
        #Average out contribution of each new memory token to each old memory token
        #Can choose not to average - this will just skew softmax to make larger numbers have 
        #higher probability
        oldMemTokenToNewMemAvgRelatedness /= self.embLength

        #Take softmax to normalize between 0 and 1 and get probabilities
        oldMemTokenToNewMemAvgRelatedness = softmax(oldMemTokenToNewMemAvgRelatedness)
        overallNewTokenRelatednessToOldMem = softmax(overallNewTokenRelatednessToOldMem)

        #Update each weight in knowledge graph using percent weighted relatedness of each token
        for idx, percRelated in enumerate(overallNewTokenRelatednessToOldMem):
            newMemEmbVal = newMemory[idx]
            
            #Update each new memory token's relatedness given relatedness of each old mem token scaled by 
            #overall new memory token's relatedness to old memory
            for oldMemIdx, oldMemTokenToNewMemRelatedness in enumerate(oldMemTokenToNewMemAvgRelatedness):
                oldMemSymbolIdx = oldMemory[oldMemIdx]
                self.symbolRelations[newMemEmbVal][oldMemSymbolIdx] += percRelated * oldMemTokenToNewMemRelatedness

        return

    # Update an existing memory with new values
    def UpdateExistingMemory(self, oldMemory, newMemory):
        dupMemIndices = self.checkForDuplicateMemory(oldMemory)
        if dupMemIndices != None:
            dupMemRowIdx = dupMemIndices[0]
            self.vectordb[dupMemRowIdx] = newMemory
            self.memDict[dupMemRowIdx].UseMemory()

        return
        
    def GetMatchingMemory(self, embeddedMemory):
        dupMemIndices = self.checkForDuplicateMemory(embeddedMemory)
        if dupMemIndices != None:
            dupMemIdx = dupMemIndices[0]
            return self.memDict[dupMemIdx]
        
        return None

    def GetMatchingShortTermMemory(self, embeddedMemory):
        dupeMemIndices = self.checkForDuplicateShortTermMemory(embeddedMemory)
        if dupeMemIndices != None:
            dupMemIdx = dupeMemIndices[0]
            return self.shortTermMemory[dupMemIdx]
        
        return None

    def checkForDuplicateShortTermMemory(self, inMemory):
        normalizedShortTermMem = preprocessing.normalize(self.shortTermMemory, axis=1)
        normalizedEmbMem = preprocessing.normalize(inMemory, axis=1).T
        dupMemCheck = normalizedShortTermMem @ normalizedEmbMem
        matchingMemIndices = np.where(np.isclose(dupMemCheck, 1) == True)[0]
        if len(matchingMemIndices) > 0:
            return matchingMemIndices
        
        return None

    #Returns indices in vector db of matching memory if exists, else return None for no dupe
    def checkForDuplicateMemory(self, inMemory):
        normalizedVecDB = preprocessing.normalize(self.vectordb, axis=1)
        normalizedEmbMem = preprocessing.normalize(inMemory, axis=1).T
        dupMemCheck = normalizedVecDB @ normalizedEmbMem
        matchingMemIndices = np.where(np.isclose(dupMemCheck, 1) == True)[0]
        if len(matchingMemIndices) > 0:
            return matchingMemIndices
        
        return None
    
    #Find simple number of matching tokens similarity between two memories
    def simpleSimilarity(self, oldMemory, newMemory):
        return len(np.intersect1d(oldMemory, newMemory))

class Memory:
    #embMemVec is embedding of symbols associated with memory
    def __init__(self, embMemVec, vocab):
        self.activation = 0
        self.embMemVec = embMemVec.flatten()

        #Convert embedded vector into symbols
        self.symbolStream = np.empty((len(self.embMemVec)), dtype=object)
        for idx, val in enumerate(self.embMemVec):
            self.symbolStream[idx] = vocab.GetSymbol(val)

        self.firstPresentation = time.time()
        #NOTE - Assumes that memory module is the only one creating the memory (if this changes, have to adjust)
        self.numPresentations = 1
    
    #Use the memory chunk and increase its activation level
    #decay is decay param in ACT-R activation equation
    #mismatchPenalty is mismatch penalty in ACT-R activation equation
    #noise is added noise in activation equation
    def UseMemory(self, decay=HyperParams["decay"], decayRate=HyperParams["decayRate"], mismatchPenalty=HyperParams["mismatchPenalty"]):
        self.numPresentations += 1
        self.activation = self.GetBaseActivation() + self.CalculateNoiseVal(HyperParams["noiseMean"], HyperParams["noiseVariance"])
        return
    
    def GetBaseActivation(self, decay=HyperParams["decay"], decayRate=HyperParams["decayRate"]):
        timeDiff = time.time() - self.firstPresentation
        if timeDiff > 0:
            return math.log(self.numPresentations/(1-decay)) - decayRate*decay*math.log(timeDiff)
        
        return 0
    
    def GetSpreadingActivation(self, embMemVecToCompare):
        matchingSlots = np.in1d(self.embMemVec, embMemVecToCompare)
        normalizedSpreadingAct = np.sum(matchingSlots) / len(self.embMemVec)
        return normalizedSpreadingAct
    
    #Technically random.uniform(a,b) returns [a,b) but accurate enough
    def CalculateNoiseVal(self, mean, variance):
        return random.uniform(mean-variance, mean+variance)
    
    #TODO - with adjacency matrix in mem module keeping track of graphical relations, may not need this
    def GetContextualActivation(self, embMemVecToCompare):
        return self.activation + self.GetSpreadingActivation(embMemVecToCompare)
    
    def GetSymbolStream(self):
        return self.symbolStream

'''Tests
test = MemoryModule(10)
test.AddEmbeddedMemory(np.ones((1,10)))
test.printVecDB()
testMemory = Memory('iron-gold')
time.sleep(0.25)
testMemory.UseMemory()
time.sleep(15)
print(testMemory.GetBaseActivation())

test = np.arange(0,6).reshape(3,2).astype(np.float64)
memMod = MemoryModule(2)

memMod.AddEmbeddedMemory(np.array([test[0]]))
memMod.AddEmbeddedMemory(np.array([test[1]]))
memMod.AddEmbeddedMemory(np.array([test[2]]))

toCompare = np.arange(0,2).reshape(1,2).astype(np.float64)
print(test)
print(toCompare)
print(memMod.TopKSimilarVectors(toCompare, 2))

#Some spreading activation tests

vocab = SymbolVocabulary()
vocab.AddSymbol("I")
vocab.AddSymbol("am")
vocab.AddSymbol("groot")

embVec = np.empty((1,3))

embVec[0][0] = vocab.GetEmbValue("groot")
embVec[0][1] = vocab.GetEmbValue("am")
embVec[0][2] = vocab.GetEmbValue("I")

memModule = MemoryModule(3, vocab)
memModule.AddEmbeddedMemory(embVec)

mem1 = Memory(embVec, vocab)
mem2 = Memory(np.ones((1,3), dtype=float), vocab)
mem1.GetSpreadingActivation(mem2.embMemVec)

#Test code for adjacency matrix creation and weighting memories by it
adjMatrix = np.arange(1,10).reshape(3,3)
storedMemories = np.array([[0,2],[1,1]])
toCompareMemory = np.array([1,2])
print(adjMatrix)

contextuallyWeightedVecDB = np.zeros((len(toCompareMemory), storedMemories.shape[0]))
print(contextuallyWeightedVecDB.shape)

for idx, weights in enumerate(adjMatrix[toCompareMemory]):
    print(weights[storedMemories[:,idx]])
    contextuallyWeightedVecDB[idx] = weights[storedMemories[:,idx]]

print(contextuallyWeightedVecDB)
print(contextuallyWeightedVecDB.T)
contextuallyWeightedVecDB = contextuallyWeightedVecDB.T

print(np.sum(contextuallyWeightedVecDB, axis=1))

#Test code for sample trades and top-k similarity comparison
vocab = SymbolVocabulary()
vocabSample = ["iron", "stone", "gold", "aluminum", "horses", "success", "fail", "undetermined", "1", "2", "player1", "aiplayer"]

for word in vocabSample:
    vocab.AddSymbol(word)

trade1 = ["player1", "1", "iron", "2", "stone", "aiplayer", "success", "1"]
trade2 = ["aiplayer", "2", "stone", "1", "iron", "player1", "fail", "undetermined"]
trade3 = ["player1", "2", "stone", "1", "iron", "aiplayer", "success", "2"]

toCompTrade = ["player1", "2", "iron", "1", "stone", "aiplayer", "success", "2"]

memMod = MemoryModule(vocab)

embTrade1 = memMod.CreateEmbMemVec(trade1)
print(embTrade1)
memMod.AddEmbeddedMemory(embTrade1)
memMod.AddEmbeddedMemory(embTrade1) #duplicate add test

embTrade2 = memMod.CreateEmbMemVec(trade2)
print(embTrade2)
memMod.AddEmbeddedMemory(embTrade2)

embTrade3 = memMod.CreateEmbMemVec(trade3)
print(embTrade3)
memMod.AddEmbeddedMemory(embTrade3)

embToComp = memMod.CreateEmbMemVec(toCompTrade)

print("Emb Trade 1 Comp:")
print(np.intersect1d(embToComp, embTrade1))
print(len(np.intersect1d(embToComp, embTrade1)))

print("Emb Trade 2 Comp:")
print(np.intersect1d(embToComp, embTrade2))
print(len(np.intersect1d(embToComp, embTrade2)))

print("Emb Trade 3 Comp:")
print(np.intersect1d(embToComp, embTrade3))
print(len(np.intersect1d(embToComp, embTrade3)))

print(memMod.TopKSimilarVectors(embToComp, 2))

memMod.adjustKnowledgeGraphSimilarity(embTrade1, embToComp)
memMod.UpdateMemoryGraph(embToComp)
'''