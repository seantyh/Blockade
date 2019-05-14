import pickle
from pathlib import Path
import numpy as np

class CwnNodeVec:
    cache_fname = "cwn_node_vec_{suffix}.pkl"
    cache_dir = Path(__file__).parent / "../../resources/"
    def __init__(self, name, **kwargs):
        self.name = name
        self.embed = None
        self.itos = None
        self.stoi = None
        self.load_cache()            

    def load_cache(self):
        cache_dir = CwnNodeVec.cache_dir
        cache_path = cache_dir / CwnNodeVec.cache_fname.format(suffix=self.name)
        with open(cache_path, "rb") as fin:
            self.embed, self.stoi, self.itos = pickle.load(fin)            
        print("load CwnNodeVec from cache: ", cache_path)   

    def vectors(self):
        itos = self.itos        
        wv = self.embed
        vectors = []
        for idx in range(len(itos)):
            vec_x = wv.get_vector(str(idx))
            vectors.append(vec_x)
        return np.vstack(vectors)

    def node_most_similar(self, word:str, charonly=True, topn=5):    
        itos = self.itos
        stoi = self.stoi
        wv = self.embed
        candids = wv.most_similar(str(stoi[word]), topn=100)
        ret = []
        for candid_i, val in candids:        
            candid_s = itos[int(candid_i)]
            if charonly and len(candid_s) > 1:
                continue
            ret.append((candid_s, val))
            if len(ret) >= topn:
                break
        return ret
    