from player import Player
from GameMode import Turn
from memoryModule import MemoryModule
from negotiator import NegotiatorWrapper as ngt
import numpy as np
import random
import math

class AIPlayer(Player):
    def __init__(self):
        super().__init__()
        self.playerOffer = None
        self.symbolVocab = None
        self.memoryModule = None
        self.ngtWrapper = ngt()

        #Algorithmic Parameters
        self.highAssociationColsToSearch = 2
        return
    
    def Update(self):
        #Print the AI's values to terminal for debug view of how it values items
        self.DebugPrintResourceValueMap()

        currentTurnType = self.gameMode.WhoseTurnIsIt()
        turnRoundStartingPlayer = self.gameMode.GetTurnRoundStartingPlayer()
        if currentTurnType == Turn.TURNEND:
            #If we started negotiating with the player first then clear player offer when turn ends
            if turnRoundStartingPlayer == Turn.PLAYER1:
                self.ClearPlayerOffer()
        elif currentTurnType == Turn.AIPLAYER:
            print("AI Player update")
            if not self.playerOffer:
                #If we're starting the negotiation round and no player offer is registered yet, kick off negotiation
                if turnRoundStartingPlayer == Turn.AIPLAYER:
                    amtItemToOffer, itemToOffer, amtItemToGet, itemToGet = self.EvaluateOffer()
                    trade = "AI Player: \nI'll trade you " + str(amtItemToOffer) + " " + itemToOffer + " for " + str(amtItemToGet) + " " + itemToGet
                    llmTrade = self.ngtWrapper.llm("Make a trade offer to player1 of " + str(amtItemToOffer) + " " + str(itemToOffer) + "in return for " + str(amtItemToGet) + str(itemToGet))
                    self.gameInstance.SetOfferOnTable((amtItemToGet, itemToGet, amtItemToOffer, itemToOffer))
                    app = self.gameInstance.GetAppReference()
                    app.UpdateTradeHistoryText(llmTrade)
                    app.LogConversation(llmTrade)
                    print(trade)
                return
            
            item1Amt,item1,item2Amt,item2 = self.playerOffer
            noItemIsNull = item1Amt != None and item1 != None and item2Amt != None and item2 != None

            if noItemIsNull:
                #symbol stream to feed to memory module
                symbolStream = ["player1", item1, item1Amt, "aiplayer", item2, item2Amt, "undetermined", "undetermined"]
                embMemVec = self.memoryModule.CreateEmbMemVec(symbolStream)

                #Find best matching memory (highest activation of topk similar approach)
                topkSimilarMemories = self.memoryModule.TopKSimilarVectors(embMemVec)
                if topkSimilarMemories.shape[0] != 0:
                    highestAct = -float("inf")
                    bestMemoryMatch = None
                    for embMem in topkSimilarMemories:
                        embMem2d = embMem.reshape(1,len(embMem))
                        mem = self.memoryModule.GetMatchingMemory(embMem2d)
                        memAct = mem.GetBaseActivation()
                        
                        if memAct != None and memAct > highestAct:
                            highestAct = memAct
                            bestMemoryMatch = mem

                    #print it out so I can see
                    print(bestMemoryMatch.GetSymbolStream())

                #First add new memory to short-term, then update memory graph
                self.memoryModule.AddToShortTermMemory(embMemVec)
                self.memoryModule.UpdateMemoryGraph(embMemVec, topkSimilarMemories)

                aiHasItem2,aiItem2Stats = self.HasResource(item2)
                player1 = self.gameInstance.GetHumanPlayer()
                playerHasItem1,playerItem1Stats = player1.HasResource(item1)

                pItem1ResCnt = 0

                if not playerHasItem1:
                    print("Are you trying to fool me? I don't do business with scoundrels.")
                    return
                else:
                    pItem1ResCnt = playerItem1Stats.GetCount()
                    if (pItem1ResCnt < item1Amt):
                        print("Are you trying to fool me? I don't do business with scoundrels.")
                        return

                if (aiHasItem2):
                    resourceAmt = aiItem2Stats.GetCount()
                    resourceVal = aiItem2Stats.GetValue()

                    totalAILoss = item2Amt * resourceVal
                    totalAIGain = item1Amt * self.GetResourceValue(item1)

                    if (resourceAmt < item2Amt):
                        print("Sorry I only have " + str(resourceAmt) + " " + item2)
                        return

                    if totalAIGain > totalAILoss:
                        #Calculate net gain and add to AI score
                        netAIGain = totalAIGain - totalAILoss
                        self.gameMode.IncrementAIScore(netAIGain)
                        
                        playerLoss = item1Amt * player1.GetResourceValue(item1)
                        playerGain = item2Amt * player1.GetResourceValue(item2)
                        netPlayerGain = playerGain - playerLoss
                        self.gameMode.IncrementPlayerScore(netPlayerGain)

                        self.UpdateResourceCount(item2, -item2Amt)
                        self.UpdateResourceCount(item1, item1Amt)

                        player1.UpdateResourceCount(item2, item2Amt)
                        player1.UpdateResourceCount(item1, -item1Amt)
                        
                        #Find memory in short-tern memory and update
                        #Then clear short-term memory (adding short term memories to long-term)
                        shortTermMem = self.memoryModule.GetMatchingShortTermMemory(embMemVec)
                        shortTermMem[-1] = self.symbolVocab.GetEmbValue(str(netAIGain))
                        shortTermMem[-2] = self.symbolVocab.GetEmbValue("success")
                        self.memoryModule.ClearShortTermMemory()
                        
                        #Signal end of negotiation round
                        self.gameMode.EndNegotiationRound()
                        #clear stored last parsed player trade
                        self.ClearPlayerOffer()
                        self.gameInstance.ClearOfferOnTable()

                        acceptanceText = "I accept this offer."
                        app = self.gameInstance.GetAppReference()
                        app.UpdateTradeHistoryText(acceptanceText)
                        app.UpdateResourcesText()
                        app.LogConversation("AI Player:\n" + acceptanceText)
                        app.LogConversation("<New Negotiation Round>")
                        print(acceptanceText)
                        
                    else:
                        #Figure out a counter offer to propose
                        amtItemToOffer, itemToOffer, amtItemToGet, itemToGet = self.EvaluateOffer()
                        self.gameInstance.SetOfferOnTable((amtItemToGet, itemToGet, amtItemToOffer, itemToOffer))

                        trade = "AI Player: \nI'll trade you " + str(amtItemToOffer) + " " + itemToOffer + " for " + str(amtItemToGet) + " " + itemToGet

                        llmTrade = self.ngtWrapper.llm("Make a trade offer to player1 of " + str(amtItemToOffer) + " " + str(itemToOffer) + "in return for " + str(amtItemToGet) + str(itemToGet))
                        app = self.gameInstance.GetAppReference()
                        app.UpdateTradeHistoryText(llmTrade)
                        app.LogConversation(llmTrade)

                        print("Simple print counter-offer:")
                        print(trade)
                        print("")

                        #output = self.ngtWrapper.llm("Make a trade offer to player1 of " + str(amtItemToOffer) + " " + str(itemToOffer) + "in return for " + str(amtItemToGet) + str(itemToGet))
                        #print(output)
            
        return
    
    #Checks memory database for similar memories that are high association
    #Returns all matches
    def CheckMemoryForOffers(self, embMemVec, highAssocationItems):
        #Find best matching memory (highest activation of topk similar approach)
        topkSimilarMemories = self.memoryModule.TopKSimilarVectors(embMemVec)

        memoriesToCheck = []
        for topkMemory in topkSimilarMemories:
            topkItemToGet = self.symbolVocab.GetSymbol(topkMemory[1])
            topkItemToGive = self.symbolVocab.GetSymbol(topkMemory[4])

            if topkItemToGet in highAssocationItems or topkItemToGive in highAssocationItems:
                memoriesToCheck.append(topkMemory)

        return memoriesToCheck
    
    def EvaluateOffer(self):
        #I can say here get what I think is highly associated and will give me a positive return
        #If it doesn't give me a positive return like I expect, decrease the association between the two things
        sortedMemoryGraphItems = self.memoryModule.GetItemRelationsSortedMemoryGraph(1, self.gameInstance.NUM_GAME_RESOURCES)
        sortedItemValues = self.memoryModule.GetItemNamesFromEmbeddings(1, self.gameInstance.NUM_GAME_RESOURCES, sortedMemoryGraphItems)
        uniqueVals = np.unique(sortedItemValues[:,-self.highAssociationColsToSearch:])

        humanPlayer = self.gameInstance.GetHumanPlayer()

        highAssociationItemsAIOwns = []
        highAssociationItemsAIDoesntOwn = []
        for val in uniqueVals:
            if self.HasResource(val)[0]:
                highAssociationItemsAIOwns.append(val)
            else:
                highAssociationItemsAIDoesntOwn.append(val)

        print("association values AI owns:")
        print(highAssociationItemsAIOwns)

        print("Association values AI doesn't own:")
        print(highAssociationItemsAIDoesntOwn)

        amtItemToOffer, itemToOffer, amtItemToGet, itemToGet = self.ChooseItemsAndAmounts(sortedItemValues, highAssociationItemsAIOwns, highAssociationItemsAIDoesntOwn)        
        symbolStream = ["player1", itemToGet, amtItemToGet, "aiplayer", itemToOffer, amtItemToOffer, "undetermined", "undetermined"]
        embMemVec = self.memoryModule.CreateEmbMemVec(symbolStream)
        relevantMemories = self.CheckMemoryForOffers(embMemVec, uniqueVals)

        #TODO - check expected value of current trade and compare against previous trades expected value
        #Prefer trade with higher expected value
        #If player response doesn't give expected value modify memory graph association values
        
        #First seed the high association items with the previously selected items 
        highAssociationItemsAIOwns = [itemToOffer]
        highAssociationItemsAIDoesntOwn = [itemToGet]

        #Check topk memories for other items to consider
        for relevantMem in relevantMemories:
            rewardAmt = self.symbolVocab.GetSymbol(relevantMem[7])
            if rewardAmt != "undetermined":
                if int(rewardAmt) > 0:
                    #TODO - update trade to offer to this one instead and update currentTradeVal
                    tempItemToGive = self.symbolVocab.GetSymbol(relevantMem[4])
                    tempItemToGet = self.symbolVocab.GetSymbol(relevantMem[1])

                    if self.HasResource(tempItemToGive):
                        highAssociationItemsAIOwns.append(tempItemToGive)

                    if humanPlayer.HasResource(tempItemToGet):
                        highAssociationItemsAIDoesntOwn.append(tempItemToGet)
                    
                    continue

        #Redo selection with topk memory values associated with previous selection
        #Return an offer using these topk memory values and high association values
        amtItemToOffer, itemToOffer, amtItemToGet, itemToGet = self.ChooseItemsAndAmounts(sortedItemValues, highAssociationItemsAIOwns, highAssociationItemsAIDoesntOwn)

        return (amtItemToOffer, itemToOffer, amtItemToGet, itemToGet)
    
    def ChooseItemsAndAmounts(self, sortedItemValues, highAssociationItemsAIOwns, highAssociationItemsAIDoesntOwn):
        gameResources = self.gameInstance.GetGameResources()
        humanPlayer = self.gameInstance.GetHumanPlayer()

        itemToGet = None
        itemToOffer = None
        amtItemToGet = 0
        amtItemToOffer = 0

        ownedHighAssociationItemIndices = np.arange(len(highAssociationItemsAIOwns))
        np.random.shuffle(ownedHighAssociationItemIndices)

        notOwnedHighAssociationItemIndices = np.arange(len(highAssociationItemsAIDoesntOwn))
        np.random.shuffle(notOwnedHighAssociationItemIndices)

        #TODO - rather than if/elif/else check here break out the core item finding code into function
        if len(highAssociationItemsAIDoesntOwn) == 0:
            #choose a high association item I own and get a high association value corresponding to it that the player owns
            for ownedItemIdx in ownedHighAssociationItemIndices:
                itemToOffer = highAssociationItemsAIOwns[ownedItemIdx]        
                memoryGraphRowToSearch = np.where(gameResources == itemToOffer)[0]
                for idx in range(self.gameInstance.NUM_GAME_RESOURCES-1, -1, -1):
                    oppItemToCheck = sortedItemValues[memoryGraphRowToSearch,idx][0]
                    #TODO - this is to bias it away from offering A for A, but could cause a crash if 
                    #player only has item A left
                    if humanPlayer.HasResource(oppItemToCheck)[0] and oppItemToCheck != itemToOffer:
                        amtItemToOffer, amtItemToGet, validTrade = self.ChooseAmtItemsToOffer(humanPlayer, itemToOffer, oppItemToCheck)
                        #If items selected are not available in quantity desired by AI, choose another item
                        if not validTrade:
                            continue

                        print(oppItemToCheck)
                        itemToGet = oppItemToCheck
                        break
                #We found an item to trade for so stop looping
                if itemToGet != None:
                    break
        elif len(highAssociationItemsAIOwns) == 0:
            #choose a high association item the player owns and get a high association value corresponding to it that I own
            for notOwnedItemIdx in notOwnedHighAssociationItemIndices:
                itemToGet = highAssociationItemsAIDoesntOwn[notOwnedItemIdx]
                memoryGraphRowToSearch = np.where(gameResources == itemToGet)[0]
                for idx in range(self.gameInstance.NUM_GAME_RESOURCES-1, -1, -1):
                    ownItemToCheck = sortedItemValues[memoryGraphRowToSearch,idx][0]
                    #TODO - this is to bias it away from offering A for A, but could cause a crash if 
                    #player only has item A left
                    if humanPlayer.HasResource(ownItemToCheck)[0] and ownItemToCheck != itemToOffer:
                        amtItemToOffer, amtItemToGet, validTrade = self.ChooseAmtItemsToOffer(humanPlayer, ownItemToCheck, itemToGet)
                        #If items selected are not available in quantity desired by AI, choose another item
                        if not validTrade:
                            continue
                        
                        print(ownItemToCheck)
                        itemToOffer = ownItemToCheck
                        break
                if itemToOffer != None:
                    break
            print("no AI owned items selected to trade for")

        else:
            for ownedItemIdx in ownedHighAssociationItemIndices:
                for notOwnedItemIdx in notOwnedHighAssociationItemIndices:
                    itemToOffer = highAssociationItemsAIOwns[ownedItemIdx]
                    itemToGet = highAssociationItemsAIDoesntOwn[notOwnedItemIdx]

                    amtItemToOffer, amtItemToGet, validTrade = self.ChooseAmtItemsToOffer(humanPlayer, itemToOffer, itemToGet)
                    if validTrade:
                        break
                if validTrade:
                    break
            
            #TODO - refactor this so it occurs in all if/else conditions above and not just here
                #   also should put this in a separate function to clean things up
                    #If we still fail and don't have any items to trade, keep searching but one column down for some valid trade
            lastColumnSearched = self.highAssociationColsToSearch
            while not validTrade:
                nextHighAssociationColumn = sortedItemValues[:, -(lastColumnSearched+1):-lastColumnSearched]
                uniqueVals = np.unique(nextHighAssociationColumn)
                
                for val in uniqueVals:
                    if self.HasResource(val)[0]:
                        highAssociationItemsAIOwns.append(val)
                    else:
                        highAssociationItemsAIDoesntOwn.append(val)

                for ownedItem in highAssociationItemsAIOwns:
                    for notOwnedItem in highAssociationItemsAIDoesntOwn:
                        itemToOffer = ownedItem
                        itemToGet = notOwnedItem

                        amtItemToOffer, amtItemToGet, validTrade = self.ChooseAmtItemsToOffer(humanPlayer, itemToOffer, itemToGet)
                        if validTrade:
                            break
                    if validTrade:
                        break

                lastColumnSearched += 1

        #TODO - add a check in case we didn't find a valid trade
        
        return amtItemToOffer, itemToOffer, amtItemToGet, itemToGet

    def ChooseAmtItemsToOffer(self, humanPlayer, itemToTrade, itemToGet):
        amtItemToTrade = -1
        amtItemToGet = -1
        validOffer = False
        
        ownItemVal = self.GetResourceValue(itemToTrade)
        oppItemVal = self.GetResourceValue(itemToGet)

        timesGreater = 0

        if ownItemVal > oppItemVal:
            timesGreater = ownItemVal / oppItemVal
        else:
            #If the opponent's item is valued greater than my own, offer 
            #trade anywhere in range of the number of items they have available
            timesGreater = oppItemVal /ownItemVal 
            playerHasItem1,playerItemToGetStats = humanPlayer.HasResource(itemToGet)
            cntItemToGet = playerItemToGetStats.GetCount()

            upperRangeToOffer = cntItemToGet if cntItemToGet < timesGreater else timesGreater
            amtItemToGet = random.randrange(1,upperRangeToOffer+1)
            amtItemToTrade = 1
            validOffer = True
            return (amtItemToTrade, amtItemToGet, validOffer)

        
        timesGreater = math.ceil(timesGreater)

        playerHasItem1,playerItemToGetStats = humanPlayer.HasResource(itemToGet)
        cntItemToGet = playerItemToGetStats.GetCount()

        #If ai is trying to get more items than player owns, find another offer
        if cntItemToGet > timesGreater:
            amtItemToGet = timesGreater
            amtItemToTrade = 1
            validOffer = True

        return (amtItemToTrade, amtItemToGet, validOffer)

    def CheckInternalState(self):
        aiScore = self.gameMode.GetAIScore()
        playerScore = self.gameMode.GetPlayerScore()

        if playerScore > aiScore:
            #More exploitative - unless at threshold where AI is close to losing; then take risks
            return
        if aiScore > playerScore:
            #More explorative - unless it's a close score in which case can be more exploitative
            #also if have a trade which is very successful can simply offer that again (exploit)
            return
        #TODO - consider, is it better to demarkate state based on known history of trades rather than just score?

    def GetTradeValue(self, amtToTrade, itemToTrade, amtToGet, itemToGet):
        itemToTradeVal = self.GetResourceValue(itemToTrade)
        itemToGetVal = self.GetResourceValue(itemToGet)

        return amtToGet * itemToGetVal - amtToTrade * itemToTradeVal

    def InitializeMemoryModule(self, symbolVocab):
        self.symbolVocab = symbolVocab

        #Add trade eval symbols to game symbol vocab
        TRADE_EVAL_SYMBOLS = ["success", "fail", "undetermined"]
        for symbol in TRADE_EVAL_SYMBOLS: 
            symbolVocab.AddSymbol(symbol)

        #Add standard set of numbers to vocab
        for num in range(1,11):
            symbolVocab.AddSymbol(str(num))

        self.memoryModule = MemoryModule(symbolVocab)

    def SaveMemoryGraph(self):
        if not self.gameMode:
            return
        
        roundNum = self.gameMode.GetRoundNumber()
        fileName = "MemoryGraphRound" + str(roundNum)
        gameResources = self.gameInstance.GetGameResources()
        self.memoryModule.PlotMemoryGraph(fileName, gameResources, title="Token Relatedness in Memory Module Round "+str(roundNum))
    
    #Clears last player offer (done if we've finished successfully negotiating a trade)
    def ClearPlayerOffer(self):
        self.playerOffer = None

    def GetPlayerOffer(self):
        return self.playerOffer

    #Assumes offer is a valid tuple
    def UpdatePlayerOffer(self, offer):
        self.playerOffer = offer

    def PlotMemoryGraph(self, graphName, xYLabels):
        self.memoryModule.PlotMemoryGraph(graphName, xYLabels)