import pyactr as actr
import numpy as np
import csv
from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import re
from collections import Counter

#Enter your OpenAI API Key here to play game with LLM output
API_KEY = ""

class NegotiatorWrapper:
    def __init__(self, apiKey=API_KEY):
        self.llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=apiKey)
        self.memory = ConversationKGMemory(llm=self.llm)
        self.trades = {}
        return
    
    def SaveMemory(self, inputMemoryText="", outputMemoryText=""):
        if inputMemoryText != "" or outputMemoryText != "":
            self.memory.save_context({"input": inputMemoryText}, {"output" : outputMemoryText})

    def GetTradesFromMemory(self):
        exchangeHistory = self.memory.load_memory_variables({"input": "What will Player1 exchange?"})["history"]

        historyLength = len(exchangeHistory)
        seperatedTrades = []
        for match in re.finditer('exchange', exchangeHistory):
            tradeSymbolIdx = match.start() + len("exchange ") 
            firstPeriodIdx = exchangeHistory.find(".", tradeSymbolIdx, historyLength)
            symbol = exchangeHistory[tradeSymbolIdx:firstPeriodIdx]
            freq = Counter(symbol)
            if freq['-'] == 1:
                symbol += "-" + symbol
            seperatedTrades.append(symbol)
        return seperatedTrades

    #Internal use only
    def _AddToTradesDict(self, inTrades, inTradePair, inItem1Amt, inItem2Amt):
        tradeDict = inTrades[inTradePair]
        #Check if the trade amount is already registered - append to trade list if so
        if inItem1Amt in tradeDict:
            tradeDict[inItem1Amt] = np.append(tradeDict[inItem1Amt],inItem2Amt)
            tradeDict[inItem1Amt] = np.sort(tradeDict[inItem1Amt])
        #Otherwise create a new entry for this trade amount
        else:
            tradeDict[inItem1Amt] = np.array([inItem2Amt])

    def AddTradeToFastMemory(self, inSeperatedTrade):
        #Handle case where condensed trade symbol into single symbol
        freq = Counter(inSeperatedTrade)
        if freq['-'] == 1:
            inSeperatedTrade += "-" + inSeperatedTrade

        tradeItems = inSeperatedTrade.split('-')
        
        item1Amt = int(tradeItems[0])
        item1 = tradeItems[1]
        item2Amt = int(tradeItems[2])
        item2 = tradeItems[3]
        
        tradePair = item1 + "-" + item2
        tradePair2 = tradeItems[3] + "-" + tradeItems[1]

        if tradePair in self.trades:
            self._AddToTradesDict(self.trades, tradePair, item1Amt, item2Amt)
            return

        if tradePair2 in self.trades:
            self._AddToTradesDict(self.trades, tradePair2, item1Amt, item2Amt)
            return

        #Otherwise, no existing table for trade type, so make one
        self.trades[tradePair] = {item1Amt : np.array([item2Amt])}

    def GetTradesForItemFromFastMemory(self, tradeSymbol):
        if tradeSymbol in self.trades:
            return self.trades[tradeSymbol]
    
        return None

    # Productions is 2d array of production name to strng
    # Chunks is dictionary of chunk names to parameters
    # Goals are array of names of goal buffers
    def CreateNegotiator(self, productionsIni, chunksIni, goalsIni):
        productions = self.ParseProductions(productionsIni)
        chunks = self.ParseChunks(chunksIni)
        goals = self.ParseGoals(goalsIni)

        negotiator = actr.ACTRModel()
        for production in productions:
            negotiator.productionstring(name=production, string=productions[production])

        for chunk in chunks:
            actr.chunktype(chunk, chunks[chunk])
        for goal in goals:
            negotiator.set_goal(goal)
        
        return negotiator

    def ParseProductions(self, productionsIni):
        lastLineNewLine = False
        productionsDict = {}
        productionString = ""
        currentProduction = ""
        with open(productionsIni, 'r') as file:
            for line in file:
                lineCopy = line.replace(" ", "")
                if (len(lineCopy) > 2 and lineCopy[-2] == ":"):
                    if lastLineNewLine:
                        productionsDict[currentProduction] = productionString
                        lastLineNewLine = False
                        productionString = ""

                    currentProduction = lineCopy[0:-2]
                    productionsDict[currentProduction] = ""
                elif lineCopy[0] == "\n":
                    if len(lineCopy) == 1:
                        lastLineNewLine = True
                else:
                    productionString += line.strip() + "\n"

        productionsDict[currentProduction] = productionString
        print(productionsDict)
        return productionsDict

    def ParseChunks(self, chunkIni):
        with open(chunkIni, 'r') as file:
            fileReader = csv.reader(file)
            chunkDict = {}
            for row in fileReader:
                slots = ""
                for item in row[1:-1]:
                    slots += item + ","
                if len(row) > 1:
                    slots += row[-1]
                chunkDict[row[0]] = slots

        return chunkDict

    def ParseGoals(self, goalsIni):
        goals = np.loadtxt(goalsIni, delimiter=",", dtype=str)
        return goals
    
    def AddToTradesDict(self, inTradePair, inItem1Amt, inItem2Amt):
        tradeDict = self.trades[inTradePair]
        #Check if the trade amount is already registered - append to trade list if so
        if inItem1Amt in tradeDict:
            tradeDict[inItem1Amt] = np.append(tradeDict[inItem1Amt],inItem2Amt)
            tradeDict[inItem1Amt] = np.sort(tradeDict[inItem1Amt])
        #Otherwise create a new entry for this trade amount
        else:
            tradeDict[inItem1Amt] = np.array([inItem2Amt])

    def ProcessTrade(self, tradeSymbol):
        tradeItems = tradeSymbol.split('-')
        
        item1Amt = int(tradeItems[0])
        item1 = tradeItems[1]
        item2Amt = int(tradeItems[2])
        item2 = tradeItems[3]
        
        tradePair = tradeItems[1] + "-" + tradeItems[3]
        tradePair2 = tradeItems[3] + "-" + tradeItems[1]

        if tradePair in self.trades:
            self.AddToTradesDict(tradePair, item1Amt, item2Amt)
            return

        if tradePair2 in self.trades:
            self.AddToTradesDict(tradePair2, item1Amt, item2Amt)
            return
        
        #Otherwise, no existing table for trade type, so make one
        self.trades[tradePair] = {item1Amt : np.array([item2Amt])}

#Debug code for testing actr agent
'''ngtWrp = NegotiatorWrapper()
ngt = ngtWrp.CreateNegotiator("../data/negotiatorAI/productionsIni.txt", "../data/NegotiatorAI/chunksIni.csv", "../data/NegotiatorAI/goalsIni.txt")

#ngt.decmem.add(actr.makechunk("", "valueEstimate", player="self", item="coal", min=2, max=4))
ngt.decmem.add(actr.makechunk("", "valueEstimate", player="opponent", item="coal", min=2, max=4))
#for i in range(2):
print(ngt.decmem)
ngt.goals["g"].add(actr.makechunk(typename="offer", item1="coal", item1amt=2, item2="iron", item2amt=3))
ngtSimulation = ngt.simulation()
#ngtSimulation.run(2)
#ngtWrp.memory.save_context({"input": "Player1 will exchange 3-horses-1-Stone. Player1 will exchange 2-horses-1-stone."}, {"output" : ""})
ngtWrp.SaveMemory({"input": "Player1 will exchange 3-horses-1-Stone. Player1 will exchange 2-horses-1-stone."}, {"output" : ""})
ngtWrp.GetTradesFromMemory()
print(ngt.decmem)
print("done")'''