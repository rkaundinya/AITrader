import numpy as np
from negotiator import NegotiatorWrapper as ngtWr
from GameInstance import GameInstance
import tkinter as tk
import os

#TODO - this should move into the actual app.py code
class App:
    def __init__(self):
        self.window = tk.Tk()
        self.gameTextFrame = tk.Frame()
        self.userInputFrame = tk.Frame()
        self.tradeHistoryFrame = tk.Frame()
        
        self.tradeInput = tk.StringVar(self.userInputFrame)
        self.userInput = tk.StringVar(self.userInputFrame)
        self.gameDataText = tk.StringVar(self.gameTextFrame)
        self.tradeHistoryText = tk.StringVar(self.tradeHistoryFrame)
        self.gameScoreText = tk.StringVar(self.tradeHistoryFrame)

        self.printMemGraphButton = tk.Button(master=self.userInputFrame, text="Save Memory Graph Image")
        self.printMemGraphButton.bind("<Button-1>", self.onPrintMemoryGraph)

        self.entry = tk.Entry(master=self.userInputFrame, textvariable=self.userInput)

        #Clear old log file then keep it open for appending
        open("logs/ConversationLog.txt", "w").close()
        self.conversationLog = open("logs/ConversationLog.txt", "a")
        return
    
    def LogConversation(self, toAdd):
        self.conversationLog.write("\n" + toAdd)
        return
    
    def CloseConversationLog(self):
        self.conversationLog.close()
        return

    def onReturn(self, event):
        print("You hit return")
        self.tradeInput.set(self.userInput.get())
        print(self.tradeInput.get())
        self.userInputReady = True

    def onEscape(self, event):
        print("You hit escape")
        self.window.destroy()

    def onPrintMemoryGraph(self, event):
        print("You saved the memory graph")
        aiPlayer = game.GetAIPlayer()
        aiPlayer.SaveMemoryGraph()

    def WaitForUserInput(self):
        self.entry.wait_variable(self.tradeInput)
        print("Done waiting")
        return self.tradeInput.get()
    
    def UpdateTradeHistoryText(self, toAdd):
        newText = self.tradeHistoryText.get() + "\n" + toAdd
        self.tradeHistoryText.set(newText)

    def UpdateGameScoreText(self):
        self.gameScoreText.set(game.GetGameScore())

    def UpdateResourcesText(self):
        self.gameDataText.set(game.uiManager.GetGameInfoText())

    def CreateNewGUI(self, game):
        #Initialize String variables with data
        self.gameDataText.set(game.uiManager.GetGameInfoText())
        self.tradeHistoryText.set("Conversation History: ")

        #Bind window events
        self.window.bind('<Return>', self.onReturn)
        self.window.bind('<Escape>', self.onEscape)

        #Create the labels
        goalLabel = tk.Label(master=self.gameTextFrame, text="Offer trades to the AI and walk away with the riches before it does!\n", anchor="center")
        resourcesLabel = tk.Label(master=self.gameTextFrame, textvariable=self.gameDataText, anchor="e", justify="left", bg="black", fg="white")
        tradeHistoryLabel = tk.Label(master=self.tradeHistoryFrame, textvariable=self.tradeHistoryText, justify="left", bg ="black", fg="white")
        gameScoreLabel = tk.Label(master=self.tradeHistoryFrame, textvariable=self.gameScoreText, justify="center", bg="black", fg="white")
        tradeInputLabel = tk.Label(master=self.userInputFrame, text="Enter your trade:")

        #place widgets into gui
        goalLabel.pack()
        resourcesLabel.pack()
        tradeHistoryLabel.pack()
        gameScoreLabel.pack()
        tradeInputLabel.pack()
        self.entry.pack()

        self.printMemGraphButton.pack()

        #render frames in gui in desired order
        self.gameTextFrame.pack(side='left')
        self.tradeHistoryFrame.pack(side='right', expand='True')
        self.userInputFrame.pack(side='bottom')

        #run the gui
        self.window.after(2000, self.RunGame)
        self.window.mainloop()

    def RunGame(self):
        while not game.ShouldQuitGame():
            game.Update()

app = App()
game = GameInstance(app)

game.NewGame()
app.CreateNewGUI(game)

print("exited while loop and game quit state is " + str(game.bQuitGame))