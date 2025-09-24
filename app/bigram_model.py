import random
from collections import defaultdict

class BigramModel:
    def __init__(self, corpus):
        self.nexts = defaultdict(list)
        for sentence in corpus:
            words = sentence.split()
            for i in range(len(words) - 1):
                self.nexts[words[i].lower()].append(words[i+1].lower())

    def generate_text(self, start_word: str, length: int) -> str:
        w = start_word.lower()
        out = [w]
        for _ in range(max(1, length - 1)):
            candidates = self.nexts.get(w)
            if not candidates:
                break
            w = random.choice(candidates)
            out.append(w)
        return " ".join(out)