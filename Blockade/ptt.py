import re
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd
from .io import load_ptt_corpus
from .utils import tqdm


class PttCorpus:
    def __init__(self, fpath):
        self.ptt_path = fpath
        self.N = None

    def getStatistics(self):
        board_counter = defaultdict(int)
        nchar_posts = defaultdict(int)
        comments_counter = defaultdict(int)
        nchar_comments = defaultdict(int)
        time_range = {}

        pat_cjk = re.compile("[\u4e00-\u9fff]")        
        def count_cjk(x):
            return len(pat_cjk.findall(x))

        n = 0
        for ent in tqdm(self.getArticles(), total=self.N):
            n += 1
            board = ent["board"]
            dt = datetime.strptime(ent["published"], "%Y-%m-%dT%H:%M:%S.000Z")
            rng_x = time_range.get(board, [datetime.now(), datetime.fromtimestamp(0)])
            rng_x[0] = dt if rng_x[0] > dt else rng_x[0]
            rng_x[1] = dt if rng_x[1] < dt else rng_x[1]
            time_range[board] = rng_x
            nchar_posts[board] += count_cjk(ent["content"])
            nchar_comments[board] += sum(count_cjk(x["content"]) for x in ent.get("comments", []))
            board_counter[board] += 1
            comments_counter[board] += len(ent.get("comments", []))
            # if n > 1000: break

        self.N = n
        n_art = pd.Series(board_counter)
        n_time_start = pd.Series({b: t0 for b, (t0, t1) in time_range.items()})
        n_time_end = pd.Series({b: t1 for b, (t0, t1) in time_range.items()})        
        ptt_info = pd.DataFrame(dict(n_article=n_art, 
                    n_comments=pd.Series(comments_counter),
                    nchar_post=pd.Series(nchar_posts),
                    nchar_comments=pd.Series(nchar_comments),
                    t0=n_time_start, t1=n_time_end))
        return ptt_info

    def getArticles(self, board=None, comments=True):
        ptt_iter = load_ptt_corpus(self.ptt_path)
        article = None
        for ent in ptt_iter:
            if "board" in ent:
                if article:
                    yield article
                if board and ent["board"] != board:
                    article = None
                    continue
                article = ent
            else:
                if not article or not comments: continue                
                if ent.get("post_id") == article["id"]:
                    article.setdefault("comments", []).append(ent)
        if article:
            yield article



