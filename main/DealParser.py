import os
import numpy as np
import torch
from EmbNet import EmbNet
from vocab import Vocab

os.environ['KMP_DUPLICATE_LIB_OK']='True'
exec(open('utilities.py').read())

class Parser:
    def __init__(self):
        data = np.load("../data/NPY_Files/NERAnnotatedTradePrompts.npy", allow_pickle=True)
        PATH = './NERClassifierNet.pth'

        #load word embeddings
        pretrainedEmbeddings = torch.load('../data/NPY_Files/embeddings.pt')

        self.embed_net = to_gpu(EmbNet(pretrainedEmbeddings))
        self.embed_net.load_state_dict(torch.load(PATH))

        self.embed_net.eval()
        self.embed_net.zero_grad()

        #Reload vocab
        self.vocab = Vocab()
        self.vocab.train(data[:,0])

    def convertString(self, sentence):
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

    def convertText(self, sample):
        out = []
        for r in sample:
            out.append(self._sample2window(r, self.vocab))
        return out

    def _sample2window(self, sample, vocab):
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

            x = self.win2v(win, vocab)
            if _x == None:
                _x = x
            else:
                _x = torch.cat((_x, x),0)
        return _x

    def win2v(self, win, vocab):
        return torch.LongTensor([vocab.encode(word) for word in win])

    def ProcessInput(self, inUserInput):
        t = self.convertString(inUserInput)
        vals = self.convertText(t)
        vals = torch.reshape(vals[0], (-1,5))
        logits = self.embed_net(to_gpu(vals))
        preds = torch.argmax(logits, dim=-1)

        preds = preds.cpu()

        #Check to make sure no repeat entities were found
        vals,entityCounts = np.unique(preds[np.where(preds != 0)[0]], return_counts=True)
        repeatEntitiesFound = len(np.where(entityCounts != 1)[0]) != 0

        bParseFailure = False
        if (repeatEntitiesFound or entityCounts.size != 4):
            bParseFailure = True
            print("Could not properly understand input")
            return bParseFailure, (None, None, None, None)

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