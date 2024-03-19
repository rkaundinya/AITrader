from player import Player
from AIPlayer import AIPlayer
from player import ResourceStats
from enum import Enum
from symbolVocabulary import SymbolVocabulary
import numpy as np

#TODO - For debugging, remove when ready for real tests
np.random.seed(1)

itemList = ["GOLD", "STONES", "WOOD", "WHEAT", "COAL", "IRON", "ALUMINUM", "HORSES"]
Resources = Enum('Resources', itemList)

def GetNumListInRange(maxVal, numItems):
    numList = []

    randInt = np.random.randint(1, maxVal-numItems+1)
    numList.append(randInt)

    sum = randInt

    while (len(numList) < numItems - 1):
        newVal = np.random.randint(1, maxVal-sum)
        numList.append(newVal)
        sum += newVal

    numList.append(maxVal-sum)
    return numList

def CreateAndInitializePlayers(inItemList=itemList):
    #Assign Resources
    player1 = Player()
    AIPlayerInstance = AIPlayer()

    #Num Items to Assign
    numItemsToAssign = 3

    inItemList = np.array(itemList)

    p1ItemIndices = np.random.randint(0, len(inItemList), numItemsToAssign)
    values,counts = np.unique(p1ItemIndices, return_counts=True)
    while (len(np.where(counts > 1)[0]) != 0):
        p1ItemIndices = np.random.randint(0, len(inItemList), numItemsToAssign)
        values,counts = np.unique(p1ItemIndices, return_counts=True)

    sharedItemIdx = np.random.choice(p1ItemIndices)

    #Do random rollouts till you choose two items that p1 doesn't have,
    #don't have repeat items, and are not choosing the shared item again
    aiItemIndices = np.random.randint(0, len(inItemList), numItemsToAssign-1)
    values,counts = np.unique(aiItemIndices, return_counts=True)
    while (np.sum(np.in1d(aiItemIndices,p1ItemIndices)) != 0\
        or len(np.where(counts > 1)[0]) != 0 or np.sum(np.in1d(aiItemIndices,sharedItemIdx)) != 0):
        aiItemIndices = np.random.randint(0, len(inItemList), numItemsToAssign-1)
        values,counts = np.unique(aiItemIndices, return_counts=True)

    aiItemIndices = np.concatenate((aiItemIndices, np.array([sharedItemIdx])), axis=0)

    #Filter out shared item from item indices
    p1ItemIndices = p1ItemIndices[np.where(p1ItemIndices != sharedItemIdx)[0]]
    aiItemIndices = aiItemIndices[np.where(aiItemIndices != sharedItemIdx)[0]]

    #Choose random values for items
    maxVal = 10

    #Assign values twice for each player's own resources
    vals = np.array(GetNumListInRange(maxVal, numItemsToAssign))
    aiVals = np.array(GetNumListInRange(maxVal, numItemsToAssign))

    counts = np.array(GetNumListInRange(maxVal, numItemsToAssign))
    aiCounts = np.array(GetNumListInRange(maxVal, numItemsToAssign))

    medCount = int(np.median(counts))
    medAICount = int(np.median(aiCounts))

    #Filter out the median from counts
    nonMedianCountIndices = np.where(counts != medCount)[0]
    nonMedianAICountIndices = np.where(aiCounts != medAICount)[0]

    counts = counts[nonMedianCountIndices]
    if counts.size < numItemsToAssign-1:
        while (counts.size < numItemsToAssign-1):
            counts = np.concatenate((counts, np.array([medCount])), axis=0)

    aiCounts = aiCounts[nonMedianAICountIndices]
    if aiCounts.size < numItemsToAssign-1:
        while (aiCounts.size < numItemsToAssign-1):
            aiCounts = np.concatenate((aiCounts, np.array([medAICount])), axis=0)

    #Sort remaining counts
    counts = np.sort(counts)
    aiCounts = np.sort(aiCounts)

    #Filter out median values and sort remaining
    medVal = int(np.median(vals))
    medAIVal = int(np.median(aiVals))

    vals = vals[np.where(vals != medVal)[0]]
    #In case there were repeat median values and we removed
    #multiple, add median values back in (ya this is messy but prototyping)
    if vals.size < numItemsToAssign-1:
        while (vals.size < numItemsToAssign-1):
            vals = np.concatenate((vals, np.array([medVal])), axis=0)
    vals = np.sort(vals)

    aiVals = aiVals[np.where(aiVals != medAIVal)[0]]
    if aiVals.size < numItemsToAssign-1:
        while (aiVals.size < numItemsToAssign-1):
            aiVals = np.concatenate((aiVals, np.array([medAIVal])), axis=0)
    aiVals = np.sort(aiVals)

    #Assign values to opposing player's resources for player
    oppVals = np.array(GetNumListInRange(maxVal-medVal, 2))
    oppVals = np.sort(oppVals)

    #How AI values opponent's resources
    aiOppVals = np.array(GetNumListInRange(maxVal-medAIVal, 2))
    aiOppVals = np.sort(aiOppVals)

    p1Resources = {}
    p1Resources[Resources(sharedItemIdx+1)] = ResourceStats(medCount, medVal)
    for idx, itemIdx in enumerate(p1ItemIndices):
        p1Resources[Resources(itemIdx+1)] = ResourceStats(counts[-1-idx], vals[idx])
        AIPlayerInstance.SetResourceValueMap(Resources(itemIdx+1), aiOppVals[-1-idx])

    player1.SetResources(p1Resources)

    aiVals = np.array(aiVals)
    aiMedVal = int(np.median(aiVals))

    aiResources = {}
    aiResources[Resources(sharedItemIdx+1)] = ResourceStats(medAICount, aiMedVal)
    for idx, itemIdx in enumerate(aiItemIndices):
        aiResources[Resources(itemIdx+1)] = ResourceStats(aiCounts[-1-idx], aiVals[idx])
        player1.SetResourceValueMap(Resources(itemIdx+1), oppVals[-1-idx])

    aiResources = dict(sorted(aiResources.items(), key=lambda item: item[1].value))
    AIPlayerInstance.SetResources(aiResources)

    player1.InitReasoningStructures()
    AIPlayerInstance.InitReasoningStructures()

    #Initialize Vocabulary
    symbolVocab = InitVocab(player1.GetGameResources())
    playerIdentifiers = ["player1", "aiplayer"]
    for playerId in playerIdentifiers:
        symbolVocab.AddSymbol(playerId)

    return player1, AIPlayerInstance, symbolVocab

def InitVocab(gameResources):
    vocab = SymbolVocabulary()
    
    for resource in gameResources:
        vocab.AddSymbol(resource)

    return vocab

def PrintAllPlayerResources(inPlayer1, inAIPlayer, bShowAIValues=False):
    print("You have these resources: ")
    inPlayer1.DebugPrintResources()

    print("\nThe AI has these resources: ")
    inAIPlayer.DebugPrintResources(bShowAIValues)