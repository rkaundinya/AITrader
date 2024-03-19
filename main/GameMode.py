'''
This file houses the core game rules and stats
Contains logic on game state and win/lose conditions
'''

from enum import Enum

class GameStats:
    def __init__(self):
        self.aiScore = 0
        self.winScore = 0
        self.loseScore = 0
        self.playerScore = 0

    def IncrementAIScore(self, increment):
        self.aiScore += increment

    def IncrementPlayerScore(self, increment):
        self.playerScore += increment

class Winner(Enum):
    PLAYER1 = 1
    AIPLAYER = 2
    DRAW = 3
    UNDETERMINED = 4

class Turn(Enum):
    TURNBEGIN = 1
    TURNEND = 2
    PLAYER1 = 3
    AIPLAYER = 4
    NONE = 5

class TurnManager:
    def __init__(self, startingPlayer=Turn.PLAYER1):
        self.currentTurn = Turn.TURNBEGIN
        self.turnOrder = {Turn.PLAYER1 : [Turn.PLAYER1, Turn.AIPLAYER], Turn.AIPLAYER : [Turn.AIPLAYER, Turn.PLAYER1]}
        self.currentRoundTurnNumber = 0
        self.roundNumber = 0
        self.startingPlayer = startingPlayer
        self.startNewRound = False

    def UpdateTurn(self):
        if self.startNewRound == True:
            self.startNewRound = False
            self.currentTurn = Turn.TURNEND
            return

        turnOrder = self.turnOrder[self.startingPlayer]
        if self.currentRoundTurnNumber < len(turnOrder):
            self.currentTurn = turnOrder[self.currentRoundTurnNumber]
            self.currentRoundTurnNumber += 1
        else:
            if self.currentTurn != Turn.TURNEND:
                self.currentTurn = Turn.TURNEND
            else:
                self.currentTurn = Turn.TURNBEGIN
                self.currentRoundTurnNumber = 0
                self.roundNumber += 1

    def SwitchTurnRoundStartPlayer(self):
        if self.startingPlayer == Turn.PLAYER1:
            self.startingPlayer = Turn.AIPLAYER
            return
        
        self.startingPlayer = Turn.PLAYER1

    def SetStartNewRound(self, bStart):
        self.currentRoundTurnNumber += 1
        self.startNewRound = bStart

class GameMode:
    def __init__(self):
        gameStats = GameStats()
        gameStats.winScore = 7
        gameStats.loseScore = -7

        self.gameStats = gameStats
        self.turnManager = TurnManager()
        return
    
    def CheckGameWinner(self):
        gameStats = self.gameStats
        if gameStats.aiScore > gameStats.winScore and gameStats.playerScore < gameStats.aiScore:
            return Winner.AIPLAYER
        if gameStats.playerScore < gameStats.loseScore:
            return Winner.AIPLAYER
        if gameStats.playerScore > gameStats.winScore and gameStats.playerScore > gameStats.aiScore:
            return Winner.PLAYER1
        if gameStats.playerScore > gameStats.winScore and gameStats.aiScore == gameStats.playerScore:
            return Winner.DRAW
        
        return Winner.UNDETERMINED
    
    def IsStartOfNewRound(self):
        return self.turnManager.currentRoundTurnNumber == 0

    def GetRoundNumber(self):
        return self.turnManager.roundNumber
    
    def EndNegotiationRound(self):
        self.turnManager.SwitchTurnRoundStartPlayer()
        self.turnManager.SetStartNewRound(True)

    def GetTurnRoundStartingPlayer(self):
        return self.turnManager.startingPlayer
    
    def IncrementAIScore(self, inc):
        self.gameStats.IncrementAIScore(inc)

    def IncrementPlayerScore(self, inc):
        self.gameStats.IncrementPlayerScore(inc)

    def Update(self):
        self.turnManager.UpdateTurn()
        print(self.WhoseTurnIsIt())
    
    def WhoseTurnIsIt(self):
        return self.turnManager.currentTurn
    
    def GetPlayerScore(self):
        return self.gameStats.playerScore
    
    def GetAIScore(self):
        return self.gameStats.aiScore