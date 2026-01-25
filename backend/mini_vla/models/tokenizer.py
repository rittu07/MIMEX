
import re

class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {
            "<pad>": 0,
            "<unk>": 1,
            "<sos>": 2,
            "<eos>": 3,
            "push": 4,
            "the": 5,
            "object": 6,
            "to": 7,
            "goal": 8,
            "pick": 9,
            "place": 10,
            "move": 11,
            "left": 12,
            "right": 13,
            "up": 14,
            "down": 15
        }
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    def __call__(self, text):
        return self.encode(text)

    def encode(self, text, max_length=30):
        # Simple regex splitting
        words = re.findall(r"\w+", text.lower())
        tokens = [self.vocab.get(w, self.vocab["<unk>"]) for w in words]
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens += [self.vocab["<pad>"]] * (max_length - len(tokens))
        return tokens
    
    @property
    def vocab_size(self):
        return len(self.vocab)
