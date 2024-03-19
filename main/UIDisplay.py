'''
Handles displaying menu options, prompts, and other user-facing information
'''
from GameMode import Winner
from GameMode import Turn
import tkinter as tk


class UIDisplay():
    def __init__(self, gameMode, gameInstance):
        if gameMode == None:
            print("Error initializing UI display with null game instance")
            return
        
        self.gameMode = gameMode
        self.gameInstance = gameInstance

        self.window = None
        self.gameTextFrame = None
        self.userInputFrame = None
        return
        
    def printScores(self):
        playerScore = str(self.gameMode.GetPlayerScore())
        aiScore = str(self.gameMode.GetAIScore())
        print("Player Score: " + playerScore)
        print("AI Score: " + aiScore)

    
    def CheckForWinLoseDisplay(self):
        gameWinner = self.gameMode.CheckGameWinner()

        if gameWinner == Winner.UNDETERMINED:
            #print("Winner yet undetermined")
            return
        
        if gameWinner == Winner.AIPLAYER:
            print("AI has gotten the better of you, better luck next time...")
            self.printScores()
            print("Game Over.")
        if gameWinner == Winner.PLAYER1:
            print("You have walked away with the riches of the automaton. Your glory will never be forgotten")
            self.printScores()
            print("Game Over.")
        if gameWinner == Winner.DRAW:
            print("You and the AI have managed to walk away on equal terms. Your cooperation sets a precedent for generations to come.")
            self.printScores()
            print("Game Over.")

        #If winner wasn't undetermined we know the winner so start a new game
        self.gameInstance.NewGame()

    def ShowNewGameInfo(self):
        print("Your job is to convince the AI to make a deal such that you get the best value by offering it trades")
        humanPlayer = self.gameInstance.GetHumanPlayer()
        print("You have these resources: ")
        humanPlayer.DebugPrintResources()
        aiPlayer = self.gameInstance.GetAIPlayer()
        print("The AI has these resources: ")
        aiPlayer.DebugPrintResources(False)
        print("You place the following values on Resources: ")
        humanPlayer.DebugPrintResourceValueMap()

    def GetGameInfoText(self):
        result = "You have these resources: \n"
        humanPlayer = self.gameInstance.GetHumanPlayer()
        result += "\t" + humanPlayer.GetResoucesText()
        result += "The AI has these resources: \n"
        aiPlayer = self.gameInstance.GetAIPlayer()
        result += "\t" + aiPlayer.GetResoucesText(False)
        result += "You place the following values on Resources: \n"
        result += humanPlayer.GetResourceValueMapText()
        return result


    def Update(self):
        self.CheckForWinLoseDisplay()

        if (self.gameMode.WhoseTurnIsIt() == Turn.TURNEND):
            self.printScores()
        elif (self.gameMode.WhoseTurnIsIt() == Turn.TURNBEGIN):
            app = self.gameInstance.GetAppReference()
            app.UpdateGameScoreText()
            print("Type 'q' to quit 'i' to show all player inventories 'v' to show your value table")

        return