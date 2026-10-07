"""
Design Upgrade #2: Predict Motifs
Store prediction flag instead of full token ID
"""
from collections import defaultdict

class PredictiveMotifCoder:
    def __init__(self):
        self.transition = defaultdict(lambda: defaultdict(int))
        
    def train(self, token_seq):
        for a,b in zip(token_seq[:-1], token_seq[1:]):
            self.transition[a][b] += 1
            
    def most_likely_next(self, token):
        nxt = self.transition[token]
        if not nxt: return None
        return max(nxt, key=nxt.get), max(nxt.values())/sum(nxt.values())
    
    def encode_with_prediction(self, token_seq):
        encoded = []
        for i, tok in enumerate(token_seq):
            if i==0:
                encoded.append(('full', tok))
            else:
                prev = token_seq[i-1]
                best, prob = self.most_likely_next(prev)
                if best == tok and prob > 0.5:
                    encoded.append(('predicted', None))
                else:
                    encoded.append(('full', tok))
        return encoded
