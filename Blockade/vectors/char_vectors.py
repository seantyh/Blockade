from pathlib import Path
import pickle
import numpy as np
from ..io import get_data_path

class CharacterVectors:
    cache_fname = "charac_vec_{suffix}.pkl"
    cache_dir = Path(__file__).parent / "../../data/cache_charac_vec"

    @staticmethod
    def from_CwnNodeVec(node_vec, name="nodevec"):
        cvec = CharacterVectors()
        cvec.build_char_vectors(node_vec)
        cvec.post_process()
        cvec.write_cache(name)        
        return cvec

    def __init__(self, name=""):
        self.stoi = None
        self.itos = None
        self.vectors = None
        if name:
            try:
                self.load_cache(name)
            except FileNotFoundError:
                print("no cache file found")
    
    @property
    def shape(self):
        return self.vectors.shape

    def encode(self, text):
        if isinstance(text, list):
            seq = []
            SEQLEN = max(len(x) for x in text)
            PAD = self.stoi["<PAD>"]
            for text_x in text:
                seq_x = [self.stoi.get(x, self.stoi["<UNK>"]) for x in text_x]    
                seq_x.extend([PAD] * (SEQLEN-len(text_x)))
                seq.append(seq_x)
            return np.vstack(seq)
        else:
            seq = [self.stoi.get(x, self.stoi["<UNK>"]) for x in text]
            return np.array([seq])        
    
    def decode(self, seq):
        return [self.itos.get(x, "<UNK>") for x in seq]

    def post_process(self):
        markers = ["<UNK>", "<EOS>", "<PAD>"]
        for m in markers:
            idx = len(self.stoi)
            self.stoi[m] = idx
            self.itos[idx] = m
        rs = np.random.RandomState(61235)  #pylint: disable=no-member
        self.vectors = np.concatenate([self.vectors, 
            rs.randn(len(markers)-1, self.vectors.shape[1]),
            np.zeros((1, self.vectors.shape[1]))], 0)
        
    def load_cache(self, name):
        cache_path = get_data_path(CharacterVectors.cache_dir,
                        CharacterVectors.cache_fname.format(suffix=name))
        with open(cache_path, "rb") as fin:
            self.vectors, self.stoi, self.itos = pickle.load(fin)
        print("load CharacterVectors from cache: ", cache_path)

    def write_cache(self, name):
        cache_path = get_data_path(CharacterVectors.cache_dir,
                        CharacterVectors.cache_fname.format(suffix=name))
        with open(cache_path, "wb") as fout:
            pickle.dump((self.vectors, self.stoi, self.itos), fout)
        print("write CharacterVectors to cache: ", cache_path)

    def build_char_vectors(self, node_vec):
        itos = node_vec.itos
        wv = node_vec.embed

        ctoi = {}
        vectors = []
        for idx in range(len(itos)):
            token = itos[idx]
            if len(token) > 1: continue
            ctoi[token] = len(ctoi)
            vec_x = wv.get_vector(str(idx))
            vectors.append(vec_x)
        itoc = {v: k for k, v in ctoi.items()}
        self.stoi = ctoi
        self.itos = itoc
        self.vectors = np.vstack(vectors)