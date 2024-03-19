class SymbolVocabulary:
    def __init__(self):
        self.symbolDict = {"UNKNOWN" : 0}
        self.embToSymbolDict = {0 : "UNKNOWN"}

    #Returns true if adding a new symbol, false if symbol already in vocab
    def AddSymbol(self, symbol):
        if symbol not in self.symbolDict:
            self.symbolDict[symbol] = len(self.symbolDict)
            self.embToSymbolDict[len(self.embToSymbolDict)] = symbol
            return True
        
        return False
    
    def GetEmbValue(self, symbol):
        if symbol in self.symbolDict:
            return self.symbolDict[symbol]
        
        return self.symbolDict["UNKNOWN"]
    
    def GetSymbol(self, embVal):
        if embVal in self.embToSymbolDict:
            return self.embToSymbolDict[embVal]
        
        return self.embToSymbolDict[0]

    def GetVocabSize(self):
        return len(self.symbolDict)     