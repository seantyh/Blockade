import json
import gzip
from pathlib import Path


def get_resource_path(name, filename="", create=False):
    res_dir = Path(__file__).parent / '../resources'/ name
    if not res_dir.exists():
        res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir.resolve() / filename

def get_data_path(name, filename="", create=False):
    data_dir = Path(__file__).parent / '../data'/ name
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir.resolve() / filename

def load_ptt_corpus(fpath):    
    parse_failed = 0
    with gzip.open(fpath, "rt", encoding="UTF-8") as fin:
        for ln in fin:
            try:
                ln = json.loads(ln)
                yield ln
            except Exception as ex:                
                parse_failed += 1
                continue          
    print(f"Parse failed: {parse_failed}")