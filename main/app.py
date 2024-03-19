import torch
import pyactr as actr
from EmbNet import EmbNet
from vocab import Vocab
import numpy as np
from enum import Enum
import setupGame as gameSetup
from negotiator import NegotiatorWrapper as ngt
from reasoner import Reasoner
from memoryModule import MemoryModule
import os
import random
import math

os.environ['KMP_DUPLICATE_LIB_OK']='True'
exec(open('utilities.py').read())

data = np.load("../data/NPY_Files/NERAnnotatedTradePrompts.npy", allow_pickle=True)

PATH = './NERClassifierNet.pth'

player1, AIPlayer, symbolVocab = gameSetup.CreateAndInitializePlayers()

gameResources = player1.GetGameResources()
NUM_GAME_RESOURCES = len(gameResources)
gameResourceEnum = Enum('GameResources', gameResources)

ngtWrapper = ngt()

negotiator = ngtWrapper.CreateNegotiator("../data/negotiatorAI/productionsIni.txt", "../data/negotiatorAI/chunksIni.csv", "../data/negotiatorAI/goalsIni.txt")
reasoner = Reasoner(AIPlayer, ngtWrapper.trades, negotiator)

def convertString(sentence):
  strList = []
  res = [i for j in sentence.split() for i in (j, ' ')][:-1]
  for i in res:
    if "." in i:
      strList.extend(i.split("."))
      strList[-1] = '.'
    elif "?" in i:
      strList.extend(i.split("?"))
      strList[-1] = '?'
    elif "!" in i:
      strList.extend(i.split("!"))
      strList[-1] = '!'
    else:
      strList.append(i)

  strList.append('"')
  strList.insert(0, '"')
  return np.array([strList])

def convertText(sample, vocab):
  out = []
  for r in sample:
    out.append(_sample2window(r, vocab))
  return out

def _sample2window(sample, vocab):
    _x = None
    windowSize = 2
    for ix in range(len(sample)):
        win = []
        for off in range(-windowSize, windowSize + 1):
            if 0 <= ix + off < len(sample):
                win.append(sample[ix + off].lower())
            else:
                #make a null string which we'll embed as zero vector
                win.append('')

        x = win2v(win, vocab)
        if _x == None:
          _x = x
        else:
          _x = torch.cat((_x, x),0)
    return _x

def win2v(win, vocab):
    return torch.LongTensor([vocab.encode(word) for word in win])
  
#load word embeddings
pretrainedEmbeddings = torch.load('../data/NPY_Files/embeddings.pt')

embed_net = to_gpu(EmbNet(pretrainedEmbeddings))
embed_net.load_state_dict(torch.load(PATH))

embed_net.eval()
embed_net.zero_grad()

#Reload vocab
vocab = Vocab()
vocab.train(data[:,0])

#Add trade eval symbols to game symbol vocab
TRADE_EVAL_SYMBOLS = ["success", "fail", "undetermined"]
for symbol in TRADE_EVAL_SYMBOLS: 
    symbolVocab.AddSymbol(symbol)

#Add standard set of numbers to vocab
for num in range(1,11):
    symbolVocab.AddSymbol(str(num))

memMod = MemoryModule(symbolVocab)

def PrintAllPlayerResources(bShowAIValues=False):
    print("You have these resources: ")
    player1.DebugPrintResources()

    print("\nThe AI has these resources: ")
    AIPlayer.DebugPrintResources(bShowAIValues)

def PrintScores(playerScore, aiScore):
    print("Player Score: " + str(playerScore))
    print("AI Score: " + str(aiScore))

#Game start prompts
PrintAllPlayerResources()

print("You place the following values on Resources: ")
player1.DebugPrintResourceValueMap()

print("Your job is to convince the AI to make a deal such that you get the best value by offering it trades")
print("Type 'q' to quit anytime or 'i' to show all player inventories or 'v' to show your value table")

def ProcessInput(inUserInput, inVocab, model):
    t = convertString(inUserInput)
    vals = convertText(t, inVocab)
    vals = torch.reshape(vals[0], (-1,5))
    logits = model(to_gpu(vals))
    preds = torch.argmax(logits, dim=-1)

    preds = preds.cpu()

    #Check to make sure no repeat entities were found
    vals,entityCounts = np.unique(preds[np.where(preds != 0)[0]], return_counts=True)
    repeatEntitiesFound = len(np.where(entityCounts != 1)[0]) != 0

    bParseFailure = False
    if (repeatEntitiesFound or entityCounts.size != 4):
        bParseFailure = True
        print("Could not properly understand input")
        return bParseFailure, (None, None, None, None)

    t = t.flatten()
    ner1Indices = np.where(preds == 1)[0]
    ner2Indices = np.where(preds == 2)[0]
    ner3Indices = np.where(preds == 3)[0]
    ner4Indices = np.where(preds == 4)[0]
    
    item1Amt = int(t[ner1Indices][0]) if len(ner1Indices) > 0 else None
    item1 = t[ner2Indices][0] if len(ner2Indices) > 0 else None
    item2Amt = int(t[ner3Indices][0]) if len(ner3Indices) > 0 else None
    item2 = t[ner4Indices][0] if len(ner4Indices) > 0 else None

    return bParseFailure, (item1Amt, item1.lower(), item2Amt, item2.lower())

userInput = ""

'''Game params'''
playerScore = 0
aiScore = 0
winScore = 7
loseScore = -7
#Array used to keep track of trade outcomes for opponent
yReal = None

'''Game variables'''
lowestValueResource = next(iter(AIPlayer.resources))
lowestValuedResourceIdx = AIPlayer.resourceSortedPosition[lowestValueResource]

#Game loop
while(True):
    #Some win conditions
    if aiScore > winScore and playerScore < aiScore:
        print("AI has gotten the better of you, better luck next time...")
        PrintScores(playerScore, aiScore)
        print("Game Over.")
        break
    if playerScore > winScore and playerScore > aiScore:
        print("You have walked away with the riches of the automaton. Your glory will never be forgotten")
        PrintScores(playerScore, aiScore)
        print("Game Over.")
        break
    if playerScore > winScore and aiScore == playerScore:
        print("You and the AI have managed to walk away on equal terms. Your cooperation sets a precedent for generations to come.")
        PrintScores(playerScore, aiScore)
        print("Game Over.")
        break
    
    #Some lose conditions
    if playerScore < loseScore:
        print("You have made too many ill advised trades. The bank is calling, and it's not for tea...")
        PrintScores(playerScore, aiScore)
        print("Game Over.")
        break

    userInput = input("Enter your deal: ")
    #userInput = "I will trade you 1 iron for 1 iron."
    if (userInput == "q"):
        break
    if (userInput.lower() == "i"):
        PrintAllPlayerResources()
        continue
    if (userInput.lower() == "v"):
        print("You place the following values on Resources:")
        player1.DebugPrintResourceValueMap()
        continue

    #Process deal with NLP
    parseFailure,items = ProcessInput(userInput, vocab, embed_net)

    #Handle AI lack of comprehension
    if parseFailure:
        print("I didn't catch that, can you phrase your offer differently?")
        continue

    item1Amt,item1,item2Amt,item2 = items

    noItemIsNull = item1Amt != None and item1 != None and item2Amt != None and item2 != None

    #Standin Agent response logic
    if noItemIsNull:
        tradeSymbol = ""

        sameItemTrade = False

        if item1 == item2:
            sameItemTrade = True

        if sameItemTrade and item1Amt == item2Amt:
            tradeSymbol = str(item1Amt) + "-" + item1
        else:
            tradeSymbol = str(items[0]) + "-" + items[1] + "-" + str(items[2]) + "-" + items[3]
        

        #symbol stream to feed to memory module
        symbolStream = ["player1", items[0], items[1], "aiplayer", items[2], items[3], "undetermined", "undetermined"]
        embMemVec = memMod.CreateEmbMemVec(symbolStream)

        #Find best matching memory (highest activation of topk similar approach)
        topkSimilarMemories = memMod.TopKSimilarVectors(embMemVec)
        if topkSimilarMemories.shape[0] != 0:
            highestAct = -float("inf")
            bestMemoryMatch = None
            for embMem in topkSimilarMemories:
                embMem2d = embMem.reshape(1,len(embMem))
                mem = memMod.GetMatchingMemory(embMem2d)
                memAct = mem.GetBaseActivation()
                
                if memAct != None and memAct > highestAct:
                    highestAct = memAct
                    bestMemoryMatch = mem

            #print it out so I can see
            print(bestMemoryMatch.GetSymbolStream())

        #First add new memory to short-term, then update memory graph
        memMod.AddToShortTermMemory(embMemVec)
        memMod.UpdateMemoryGraph(embMemVec, topkSimilarMemories)

        ngtWrapper.SaveMemory("Player1 will exchange " + tradeSymbol + ".")
        
        #negotiator.goals["g"].add(actr.makechunk(typename="offer", item1=item1, item1amt=item1Amt, item2=item2, item2amt=item2Amt, player="opponent"))
        negotiator.goals["g"].add(actr.makechunk(typename="offer", item1=item1, item1amt=item1Amt, item2=item2, item2amt=item2Amt, player="opponent"))
        ngtSim = negotiator.simulation()
        while True:
            ngtSim.step()
            if ngtSim.current_event.action != None:
                if ngtSim.current_event.action == "RULE FIRED: opponentEstimateKnown_Item1":
                    test = ngtSim.__buffers["retrieval"][1]
                elif ngtSim.current_event.action == "RULE FIRED: opponentEstimateKnown_Item2":
                    test = ngtSim.__buffers["retrieval"][1]
                elif ngtSim.current_event.action == "RULE FIRED: readyToReason":
                    break

        ngtWrapper.GetTradesFromMemory()

        aiHasItem2,aiItem2Stats = AIPlayer.HasResource(item2)
        playerHasItem1,playerItem1Stats = player1.HasResource(item1)

        pItem1ResCnt = 0

        if not playerHasItem1:
            print("Are you trying to fool me? I don't do business with scoundrels.")
            continue
        else:
            pItem1ResCnt = playerItem1Stats.GetCount()
            if (pItem1ResCnt < item1Amt):
                print("Are you trying to fool me? I don't do business with scoundrels.")
                continue

        if (aiHasItem2):
            resourceAmt = aiItem2Stats.GetCount()
            resourceVal = aiItem2Stats.GetValue()

            totalAILoss = item2Amt * resourceVal
            totalAIGain = item1Amt * AIPlayer.GetResourceValue(item1)

            if (resourceAmt < item2Amt):
                print("Sorry I only have " + str(resourceAmt) + " " + item2)
                continue

            if totalAIGain > totalAILoss:
                #If AI accepts the trade, add to fast memory
                ngtWrapper.AddTradeToFastMemory(tradeSymbol)

                #Calculate net gain and add to AI score
                netAIGain = totalAIGain - totalAILoss
                aiScore += netAIGain
                
                playerLoss = item1Amt * player1.GetResourceValue(item1)
                playerGain = item2Amt * player1.GetResourceValue(item2)
                netPlayerGain = playerGain - playerLoss
                playerScore += netPlayerGain

                AIPlayer.UpdateResourceCount(item2, -item2Amt)
                AIPlayer.UpdateResourceCount(item1, item1Amt)

                player1.UpdateResourceCount(item2, item2Amt)
                player1.UpdateResourceCount(item1, -item1Amt)
                
                #Find memory in short-tern memory and update
                #Then clear short-term memory (adding short term memories to long-term)
                shortTermMem = memMod.GetMatchingShortTermMemory(embMemVec)
                shortTermMem[-1] = symbolVocab.GetEmbValue(str(netAIGain))
                shortTermMem[-2] = symbolVocab.GetEmbValue("success")
                memMod.ClearShortTermMemory()
                print("I accept this offer.")
            else:
                #Figure out a counter offer to propose
                
                #I can say here get what I think is highly associated and will give me a positive return
                #If it doesn't give me a positive return like I expect, decrease the association between the two things
                sortedItemValues = memMod.GetItemRelationsSortedMemoryGraph(1, NUM_GAME_RESOURCES)
                uniqueVals = np.unique(sortedItemValues[:,-2:])
                
                highAssociationItemsAIOwns = []
                highAssociationItemsAIDoesntOwn = []
                for val in uniqueVals:
                    if AIPlayer.HasResource(val)[0]:
                        highAssociationItemsAIOwns.append(val)
                    else:
                        highAssociationItemsAIDoesntOwn.append(val)

                print("association values AI owns:")
                print(highAssociationItemsAIOwns)

                print("Association values AI doesn't own:")
                print(highAssociationItemsAIDoesntOwn)

                randHighAssOppItemIdx = random.randrange(len(highAssociationItemsAIDoesntOwn))
                itemToGet = highAssociationItemsAIDoesntOwn[randHighAssOppItemIdx]

                randHighAssOwnItemIdx = random.randrange(len(highAssociationItemsAIOwns))
                itemToOffer = highAssociationItemsAIOwns[randHighAssOwnItemIdx]

                ownItemVal = AIPlayer.GetResourceValue(itemToOffer)
                oppItemVal = AIPlayer.GetResourceValue(itemToGet)

                amtItemToGet = 0
                amtItemToOffer = 0
                timesGreater = 0

                if ownItemVal > oppItemVal:
                    timesGreater = ownItemVal / oppItemVal
                else:
                    timesGreater = oppItemVal /ownItemVal 
                
                timesGreater = math.ceil(timesGreater)

                amtItemToGet = timesGreater
                amtItemToOffer = 1

                print("Simple print counter-offer:")
                print("I'll trade you " + str(amtItemToOffer) + " " + itemToOffer + " for " + str(amtItemToGet) + " " + itemToGet)
                print("")

                '''sortedOpponentResourceTradeOffers = reasoner.GetOpponentTradeOffersDescendingOrder()

                itemToOffer = None
                itemToGet = None
                for resource in sortedOpponentResourceTradeOffers.keys():
                    if player1.HasResource(resource):
                        if lowestValuedResourceIdx > sortedOpponentResourceTradeOffers[resource]:
                            print("Best opponent tradable item worth less than least valuable AI item")
                        else:
                            itemToOffer = lowestValueResource.name.lower()
                            itemToGet = resource
                            break

                matchingChunk = reasoner.LookForPrevTradeReceivingItem(itemToOffer, "opponent")
                itemToGetAmt = 1
                if matchingChunk != None and matchingChunk['success'] != None:
                    succsessfulTrade = matchingChunk['success'].values
                    print(type(succsessfulTrade))
                    itemToGetAmt = (int)(matchingChunk['item2amt'].values)

                    if succsessfulTrade != "true" and itemToGetAmt > 1:
                        itemToGetAmt = itemToGetAmt - 1'''

                #TODO - figure out what amount to offer a trade for and input that for LLM prompt
                #output = ngtWrapper.llm("Make a trade offer to player1 of " + str(amtItemToOffer) + " " + str(itemToOffer) + "in return for " + str(amtItemToGet) + str(itemToGet))
                #print(output)

        else:
            print("Sorry I do not have any " + item2)

print("\nPlayer Value Map: ")
player1.DebugPrintResourceValueMap()

print("AI Value Map: ")
AIPlayer.DebugPrintResourceValueMap()

print("Exited Game")