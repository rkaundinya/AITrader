import torch
from EmbNet import EmbNet
from vocab import Vocab
from tqdm import tqdm
from player import Player
from player import ResourceStats
from enum import Enum
import numpy as np

exec(open('utilities.py').read())

data = np.load("../../data/NPY_Files/NERAnnotatedTradePrompts.npy", allow_pickle=True)

PATH = './NERClassifierNet.pth'

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
  
#load word embeddings
pretrainedEmbeddings = torch.load('../../data/NPY_Files/embeddings.pt')

embed_net = to_gpu(EmbNet(pretrainedEmbeddings))
embed_net.load_state_dict(torch.load(PATH))

embed_net.eval()
embed_net.zero_grad()

#Reload vocab
vocab = Vocab()
vocab.train(data[:,0])

t = convertString("I'm interested in trading 9 wheat for 11 aluminum. What do you think?")
valTest = convertText(t, vocab)
valTest = torch.reshape(valTest[0], (-1,5))
logits = embed_net(to_gpu(valTest))
preds = torch.argmax(logits, dim=-1)

print(preds)

#userStr = input("What is your offer?\n")
userStr = "Give me 5 horses for 3 gold."

t = convertString(userStr)
valTest = convertText(t, vocab)
valTest = torch.reshape(valTest[0], (-1,5))
logits = embed_net(to_gpu(valTest))
preds = torch.argmax(logits, dim=-1)

print(preds)

preds = preds.cpu()

#Check to make sure no repeat entities were found
vals,entityCounts = np.unique(preds[np.where(preds != 0)[0]], return_counts=True)
repeatEntitiesFound = len(np.where(entityCounts != 1)[0]) != 0

if (repeatEntitiesFound or entityCounts.size != 4):
    print("Could not properly understand input")

t = t.flatten()
item1Amt = int(t[np.where(preds == 1)[0]][0])
item1 = t[np.where(preds == 2)[0]][0]
item2Amt = int(t[np.where(preds == 3)[0]][0])
item2 = t[np.where(preds == 4)[0]][0]

itemList = ["GOLD", "STONES", "WOOD", "WHEAT", "COAL", "IRON", "ALUMINUM", "HORSES"]
Resources = Enum('Resources', itemList)

player = Player()
player.SetResources({Resources.GOLD : ResourceStats(2,5), Resources.WOOD : ResourceStats(3,3)})

hasResource,resourceStats = player.HasResource(item2)
resourceAmt = resourceStats.GetCount()
if (hasResource):
    if (resourceAmt < item2Amt):
        print("Sorry I only have " + str(resourceAmt) + " " + item2)
    else:
        print("I do have " + str(resourceAmt) + item2 + " to trade.")

#Assign Resources
player1 = Player()
AIPlayer = Player()

#Num Items to Assign
numItemsToAssign = 3

itemList = np.array(itemList)

p1ItemIndices = np.random.randint(0, len(itemList), numItemsToAssign)
values,counts = np.unique(p1ItemIndices, return_counts=True)
while (len(np.where(counts > 1)[0]) != 0):
    p1ItemIndices = np.random.randint(0, len(itemList), numItemsToAssign)
    values,counts = np.unique(p1ItemIndices, return_counts=True)

sharedItemIdx = np.random.choice(p1ItemIndices)

#Do random rollouts till you choose two items that p1 doesn't have,
#don't have repeat items, and are not choosing the shared item again
aiItemIndices = np.random.randint(0, len(itemList), numItemsToAssign-1)
values,counts = np.unique(aiItemIndices, return_counts=True)
while (np.sum(np.in1d(aiItemIndices,p1ItemIndices)) != 0\
    or len(np.where(counts > 1)[0]) != 0 or np.sum(np.in1d(aiItemIndices,sharedItemIdx)) != 0):
    aiItemIndices = np.random.randint(0, len(itemList), numItemsToAssign-1)
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
    AIPlayer.SetResourceValueMap(Resources(itemIdx+1), aiOppVals[-1-idx])

player1.SetResources(p1Resources)

aiVals = np.array(aiVals)
aiMedVal = int(np.median(aiVals))

aiResources = {}
aiResources[Resources(sharedItemIdx+1)] = ResourceStats(medAICount, aiMedVal)
for idx, itemIdx in enumerate(aiItemIndices):
    aiResources[Resources(itemIdx+1)] = ResourceStats(aiCounts[-1-idx], aiVals[idx])
    player1.SetResourceValueMap(Resources(itemIdx+1), oppVals[-1-idx])

AIPlayer.SetResources(aiResources)

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

#Game params
playerScore = 0
aiScore = 0
winScore = 7
loseScore = -7

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
                
                print("I accept this offer.")
            else:
                print("I'm not ready to make this trade.")

        else:
            print("Sorry I do not have any " + item2)

print("\nPlayer Value Map: ")
player1.DebugPrintResourceValueMap()

print("AI Value Map: ")
AIPlayer.DebugPrintResourceValueMap()

print("Exited Game")