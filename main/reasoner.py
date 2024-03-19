import numpy as np
import pyactr as actr

class Reasoner:
    def __init__(self, owner, fastTradeMem, actrModel):
        self.weights = None
        #Use owner to get reasoning stats
        self.playerOwner = owner
        #Use the fast trade mem struct
        self.fastTradeMem = fastTradeMem
        #Store the actrModel we're using to reason
        self.actrModel = actrModel

    #Returns offers opponent made in order of highest value item to self to lowest value
    def GetOpponentTradeOffersDescendingOrder(self):
        itemsOpponentWillTrade = {}
        for memory in self.actrModel.decmem._data.keys():
            if memory.typename == "offer":
                resourceRawName = memory.actrchunk.item1_.values
                itemsOpponentWillTrade[resourceRawName] = self.playerOwner.GetResourceSortedPosition(resourceRawName)
        itemsOpponentWillTrade = dict(sorted(itemsOpponentWillTrade.items(), key=lambda item: item[1], reverse=True))
        return itemsOpponentWillTrade
    
    #Returns highest activation trade of item opponent received
    def LookForPrevTradeReceivingItem(self, item, playerType):
        self.actrModel.goals["g"].add(actr.makechunk(typename="findOffer", item=item, player=playerType, state="begin"))
        #Debug add an offer with the item we're looking for to see if this works -- TODO remove
        self.actrModel.decmem.add(actr.makechunk("", "offer", item1="iron", item2=item, player=playerType, item1amt=2, item2amt=3, success='true'))
        ngtSim = self.actrModel.simulation()

        matchingChunk = None
        while True:
            ngtSim.step()
            if ngtSim.current_event.action != None:
                if ngtSim.current_event.action == "RULE FIRED: successfulTradeRetrievalSuccessful" or\
                     ngtSim.current_event.action == "RULE FIRED: retrieveUnsuccessfulTradeSuccessful":
                    retrievalBuffData = ngtSim._Simulation__buffers["retrieval"].dm._data
                    matchingChunk = list(retrievalBuffData.keys())[-1]._asdict()
                    break
                elif ngtSim.current_event.action == "RULE FIRED: retrieveUnsuccessfulTradeSuccessful":
                    retrievalBuffData = ngtSim._Simulation__buffers["retrieval"].dm._data
                    matchingChunk = next(iter(retrievalBuffData))
                    break
                elif ngtSim.current_event.action == "RULE FIRED: retrieveUnsuccessfulTradeFailed":
                    break

        return matchingChunk

    def LinearRegressionDirectSolution(self, observations, yReal):
        np.random.seed(1)
        adjObs = observations + (np.random.rand(observations.shape[0], observations.shape[1]) / 100)
        self.weights = np.linalg.inv(adjObs.T @ adjObs) @ adjObs.T @ yReal

    #returns weights ranked in descending order
    def GetItemsRankedByValue(self):
        return np.flip(np.argsort(self.weights[1:], axis=0), axis=0)
    
    def Predict(self, obs):
        prediction = obs[:, 1:] @ self.weights[1:] + self.weights[0][0]
        return prediction
    
#Debug Code
'''tradeData = np.array([[1,-1,0,0,1,0], [1,0,1,-1,0,0]])
yReal = np.array([[3],[2]])
reasoner = Reasoner()

reasoner.LinearRegressionDirectSolution(tradeData, yReal)
itemValsRankedByIndex = reasoner.GetItemsRankedByValue()
print(itemValsRankedByIndex)'''