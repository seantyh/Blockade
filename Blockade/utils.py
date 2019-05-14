import re
try:
    shell = get_ipython().__class__.__name__
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

def keep_cjk(text, punc=True):
    if punc:
        pat_cjk = re.compile("[\u4e00-\u9fff\uff00-\uffef\u3000-\u303f]")            
    else:
        pat_cjk = re.compile("[\u4e00-\u9fff]")            
    return "".join(pat_cjk.findall(text))

def split_sentences(text):
    return re.split("。」|：「|[。？！：]", text, )