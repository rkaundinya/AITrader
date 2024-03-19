import setupGame as gameSetup
from GameMode import GameMode
from GameMode import Winner
from UIDisplay import UIDisplay
from DealParser import Parser
import numpy as np
import tkinter as tk

class GameInstance: 
    def __init__(self, inApp=None):
        self.symbolVocab = None
        self.players = []
        self.gameResources = None
        self.offerOnTable = None
        self.gameMode = GameMode()
        self.uiManager = UIDisplay(self.gameMode, self)
        self.parser = Parser()
        self.NUM_GAME_RESOURCES = 0
        self.bQuitGame = False
        self.app = inApp
        return
    
    def GetAppReference(self):
        return self.app

    def ShouldQuitGame(self):
        return self.bQuitGame
    
    def QuitGame(self):
        self.bQuitGame = True
        self.app.CloseConversationLog()

    def resetVariables(self):
        self.players = []
        self.gameMode = GameMode()
        self.uiManager = UIDisplay(self.gameMode, self)
        self.NUM_GAME_RESOURCES = 0
        self.symbolVocab = None
        self.gameResources = None
        self.app.CloseConversationLog()

    def NewGame(self):
        self.SetupGame()
        self.uiManager.ShowNewGameInfo()
        app = self.GetAppReference()
        app.UpdateGameScoreText()
        self.gameResources = np.array(self.players[0].GetGameResources())
        self.NUM_GAME_RESOURCES = len(self.gameResources)

    def SetupGame(self):
        player1, AIPlayer, symbolVocab = gameSetup.CreateAndInitializePlayers()

        player1.SetGameMode(self.gameMode)
        AIPlayer.SetGameMode(self.gameMode)

        player1.SetGameInstance(self)
        AIPlayer.SetGameInstance(self)
        AIPlayer.InitializeMemoryModule(symbolVocab)
        
        self.players.append(player1)
        self.players.append(AIPlayer)

        self.symbolVocab = symbolVocab

    def GameOver(self):
        if self.gameMode and self.gameMode.CheckGameWinner() != Winner.UNDETERMINED:
            self.resetVariables()
            
    def ClearParsedTrades(self):
        self.players[1].ClearPlayerOffer()

    def GetParser(self):
        return self.parser

    def GetHumanPlayer(self):
        return self.players[0]
    
    def GetAIPlayer(self):
        return self.players[1]

    def GetSymbolVocab(self):
        return self.symbolVocab
    
    def GetGameResources(self):
        return self.gameResources
    
    def GetGameScore(self):
        playerScore = str(self.gameMode.GetPlayerScore())
        aiScore = str(self.gameMode.GetAIScore())

        toReturn = "Player Score: " + playerScore + "\nAI Score: " + aiScore
        return toReturn
    
    def UpdateScoreAndResources(self, player1ItemToGiveAmt, player1ItemToGive, aiItemToGiveAmt, aiItemToGive):
        aiPlayer = self.GetAIPlayer()
        humanPlayer = self.GetHumanPlayer()

        aiItemToGiveValue = aiPlayer.GetResourceValue(aiItemToGive)
        aiItemToGetValue = aiPlayer.GetResourceValue(player1ItemToGive)
        totalAILoss = aiItemToGiveValue * aiItemToGiveAmt
        totalAIGain = player1ItemToGiveAmt * aiItemToGetValue
        
        netAIGain = totalAIGain - totalAILoss
        self.gameMode.IncrementAIScore(netAIGain)
        
        playerLoss = player1ItemToGiveAmt * humanPlayer.GetResourceValue(player1ItemToGive)
        playerGain = aiItemToGiveAmt * humanPlayer.GetResourceValue(aiItemToGive)
        netPlayerGain = playerGain - playerLoss
        self.gameMode.IncrementPlayerScore(netPlayerGain)

        aiPlayer.UpdateResourceCount(aiItemToGive, -aiItemToGiveAmt)
        aiPlayer.UpdateResourceCount(player1ItemToGive, player1ItemToGiveAmt)

        humanPlayer.UpdateResourceCount(aiItemToGive, aiItemToGiveAmt)
        humanPlayer.UpdateResourceCount(player1ItemToGive, -player1ItemToGiveAmt)

    def SetOfferOnTable(self, offer):
        self.offerOnTable = offer

    def GetOfferOnTable(self):
        return self.offerOnTable

    def ClearOfferOnTable(self):
        self.offerOnTable = None

    def Update(self):
        self.uiManager.Update()

        #check if game is over
        if self.gameMode and self.gameMode.CheckGameWinner() != Winner.UNDETERMINED:
            self.resetVariables()
            self.NewGame()

        if self.gameMode and self.gameMode.IsStartOfNewRound():
            self.GetAIPlayer().SaveMemoryGraph()

        for player in self.players:
            player.Update()

        #Updates whose turn it is
        self.gameMode.Update()
        return
    
'''game = GameInstance()
game.NewGame()
window = tk.Tk()

gameTextFrame = tk.Frame()
userInputFrame = tk.Frame()

#Create string varaibles for labels
tradeInput = tk.StringVar(window)
gameDataText = tk.StringVar(gameTextFrame)

#Initialize String variables with data
gameDataText.set(game.uiManager.GetGameInfoText())

def onReturn(event):
    print("You hit return")
    print(tradeInput.get())

def onEscape(event):
    print("You hit escape")
    window.destroy()

#Bind window events
window.bind('<Return>', onReturn)
window.bind('<Escape>', onEscape)

#Create the labels
goalLabel = tk.Label(master=gameTextFrame, text="Offer trades to the AI and walk away with the riches before it does!\n", anchor="center")
resourcesLabel = tk.Label(master=gameTextFrame, textvariable=gameDataText, anchor="e", justify="left", bg="black", fg="white")
tradeInputLabel = tk.Label(master=userInputFrame, text="Enter your trade:")

#Create text input box
entry = tk.Entry(master=userInputFrame, textvariable=tradeInput)

#place widgets into gui
goalLabel.pack()
resourcesLabel.pack()
tradeInputLabel.pack()
entry.pack()

#render frames in gui in desired order
gameTextFrame.pack()
userInputFrame.pack()

#start the gui
window.mainloop()

print("Got here")'''