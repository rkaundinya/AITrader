from enum import Enum
import numpy as np
from GameMode import Turn

#Hard coded and copied from generatorTradeAndNonTrade.py for now 
#(obv would be good to clean this up)
itemList = ["GOLD", "STONES", "WOOD", "WHEAT", "COAL", "IRON", "ALUMINUM", "HORSES"]
Resources = Enum('Resources', itemList)

#Note - probably a bad way of doing this but just quick prototyping
class ResourceStats:
    def __init__(self, count, value):
        self.count = count
        self.value = value

    def UpdateCount(self, delta):
        self.count += delta

        if self.count < 0:
            self.count = 0

    def GetCount(self):
        return self.count

    def IsEmpty(self):
        return self.count <= 0

    def GetValue(self):
        return self.value

    def __str__(self):
        return "\tCount: " + str(self.count) + "\n\t" + "Value: " + str(self.value)

    

class Player:
    def __init__(self):
        #Dict of resource enum to ResourceStats obj
        self.resources = {}
        #Set of resource names for quick check
        self.resourceNames = {}
        #Map to access values of resources
        self.resourceValMap = {}
        #Map indicating position of resource value in sorted value vector
        self.resourceSortedPosition = {}
        #Vector keeping ordered track of items and values
        self.orderedResources = []
        self.gameMode = None
        self.gameInstance = None
        self.parser = None

    def Update(self):
        if self.gameMode.WhoseTurnIsIt() == Turn.PLAYER1:
            app = self.gameInstance.GetAppReference()
            userInput = app.WaitForUserInput()
            
            if (userInput.lower() == "y"):
                playerOffer = self.gameInstance.GetOfferOnTable()
                itemToGiveAmt, itemToGive, itemToGetAmt, itemToGet = playerOffer
                self.gameInstance.UpdateScoreAndResources(itemToGiveAmt, itemToGive, itemToGetAmt, itemToGet)
                self.gameMode.EndNegotiationRound()
                #TODO - this is dirty; clean this up later
                self.gameInstance.ClearParsedTrades()
                self.gameInstance.ClearOfferOnTable()

                #Clear AI Player's memory and add it to long-term
                #TODO - should handle this cleaner in the actual game update logic and
                #within AI Character itself
                aiPlayer = self.gameInstance.GetAIPlayer()
                aiPlayer.memoryModule.ClearShortTermMemory()

                app.UpdateResourcesText()
                app.LogConversation("Player accepted the AI Offer")
                app.LogConversation("<New Negotiation Round>")
                return
            
            if self.parser:
                bParseFailure, tradeSymbols = self.parser.ProcessInput(userInput)
                if (not bParseFailure):
                    self.gameInstance.GetAIPlayer().UpdatePlayerOffer(tradeSymbols)
                    app.LogConversation("Human Player:\n" + userInput)
                    print("Trade offer successfully parsed by player")
                else:
                    print("I didn't catch that, can you phrase your offer differently?")
        return

    def SetGameMode(self, inGameMode):
        self.gameMode = inGameMode

    def SetGameInstance(self, inGameInstance):
        self.gameInstance = inGameInstance
        self.parser = self.gameInstance.GetParser()

    #Used to initialize structures that assist in reasoning modules
    def InitReasoningStructures(self):
        pos = 0
        for resourcePair in sorted(self.resourceValMap.items(), key=lambda item : item[1]):
            val = resourcePair[1]
            self.orderedResources = val
            self.resourceSortedPosition[resourcePair[0]] = pos
            pos += 1
        self.orderedResources = np.array(self.orderedResources)

    def GetResourceSortedPosition(self, resourceRawName):
        resource = self.resourceNames[resourceRawName]
        if resource in self.resourceSortedPosition:
            return self.resourceSortedPosition[resource]

    #Resources expected to have key=resource enum name, val=amt
    def SetResources(self, resourcesDict):
        self.resources = resourcesDict

        for resource in self.resources.keys():
            val = self.resources[resource].GetValue()
            self.resourceNames[resource.name.lower()] = resource
            self.resourceValMap[resource] = val

    def GetGameResources(self):
        result = []
        for resourceName in self.resourceNames.keys():
            result.append(resourceName)
        return result

    def HasResource(self, resource):
        if resource in self.resourceNames:
            if self.resourceNames[resource] in self.resources:
                return True, self.resources[self.resourceNames[resource]]

        return False, -1

    def SetResourceValueMap(self, resource, value):
        self.resourceValMap[resource] = value
        self.resourceNames[resource.name.lower()] = resource

    def GetResourceValue(self, resource):
        if resource in self.resourceNames.keys():
            return self.resourceValMap[self.resourceNames[resource]]

        return None

    def UpdateResourceCount(self, resource, delta):
        resourceEnum = None

        if resource in self.resourceNames.keys():
            resourceEnum = self.resourceNames[resource]

        if resourceEnum != None:
            #Add resource to inventory if not there before
            if resourceEnum not in self.resources.keys():
                self.resources[resourceEnum] = ResourceStats(delta, self.resourceValMap[resourceEnum])
            #Otherwise, update inventory count
            else:
                self.resources[resourceEnum].UpdateCount(delta)
        else:
            print("Game Log --- Error finding resource to update")

    def GetResoucesText(self, showValue=True):
        result = ""
        for resource in self.resources.keys():
            result += "Resource: " + resource.name + "\n"
            if showValue:
                result += str(self.resources[resource]) + "\n"
            else:
                result += "\tCount: " +  str(self.resources[resource].GetCount()) + "\n"
        
        return result    

    def DebugPrintResources(self, showValue=True):
        for resource in self.resources.keys():
            print("Resource: " + resource.name)
            if showValue:
                print(self.resources[resource])
            else:
                print("\tCount: " +  str(self.resources[resource].GetCount()))

    def GetResourceValueMapText(self):
        result = ""
        for resource in self.resourceValMap.keys():
            result += "Resource: " + resource.name + ", "
            result += "Value: " + str(self.resourceValMap[resource]) + "\n"

        return result

    def DebugPrintResourceValueMap(self):
        for resource in self.resourceValMap.keys():
            print("Resource: " + resource.name, end=", ")
            print("Value: " + str(self.resourceValMap[resource]))
    

'''test = Player()
test.SetResources({Resources.GOLD : ResourceStats(5,1), Resources.STONES : ResourceStats(3,2)})
test.SetResourceValueMap(Resources.HORSES, 3)
test.UpdateResourceCount("horses", 2)
test.UpdateResourceCount("stones", 2)
test.DebugPrintResources()'''