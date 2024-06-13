import re
import collections

class Data():

    def __init__(self, fname):
        with open(fname) as f:
            self.data = f.read()

    def _preprocess(self, x=None):
        if x is not None:
           return re.sub('[^A-Za-z]+', ' ', x).lower()
        self.data = re.sub('[^A-Za-z]+', ' ', self.data).lower()

    def _tokenize(self, x=None):
        if x is not None:
            return list(x)
        self.tokens = list(self.data)

    def get_data(self):
        self._preprocess()
        self._tokenize()
        return self.data, self.tokens

    def build(self, vocab=None):
        raw_text, tokens = self.get_data()
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flattening a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [ token for line in tokens for token in line ]

        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key = lambda x: x[1], reverse=True)

        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq ])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if isinstance(indices, list):
            return [self.idx_to_token[int(index)] for index in indices ]
        return self.idx_to_token[indices]

    @property
    def unk(self):
        return self.token_to_idx['<unk>']
