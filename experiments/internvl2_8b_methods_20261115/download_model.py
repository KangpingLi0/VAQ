"""Pinned, resumable range downloads; every LFS file is SHA256 verified."""
import bootstrap
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import requests

META=json.loads((bootstrap.OUT/'model_info.json').read_text())
FILES=[x for x in META['siblings'] if '/' not in x['rfilename']
       and x['rfilename'].endswith(('.py','.json','.model','.safetensors'))]
LOCK=threading.Lock()
STATE={'status':'downloading','repository':META['id'],'revision':META['sha'],
       'required_files':len(FILES),'completed_files':[],'file_bytes':{},'model_dir':str(bootstrap.MODEL)}

def save():
    with LOCK:
        p=bootstrap.OUT/'download_status.json';q=p.with_suffix('.tmp')
        q.write_text(json.dumps(STATE,indent=2)+'\n');q.replace(p)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def fetch_range(name,start,end):
    p=bootstrap.MODEL/(name+f'.range_{start}_{end}')
    for attempt in range(10000):
        offset=start+(p.stat().st_size if p.exists() else 0)
        if offset==end+1:return p
        try:
            host='huggingface.co' if attempt%2==0 else 'hf-mirror.com'
            url=f'https://{host}/{META["id"]}/resolve/{META["sha"]}/{name}?download=true&range={offset}'
            with requests.get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(25,90)) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {offset}-{end}/'), r.headers.get('Content-Range')
                with p.open('ab') as f:
                    for chunk in r.iter_content(256*1024):f.write(chunk)
            assert p.stat().st_size==end-start+1
            return p
        except Exception as e:
            print('RETRY',name,start,attempt,type(e).__name__,str(e)[:140],flush=True)
            time.sleep(min(30,2+attempt))
    raise RuntimeError(name)

def fetch(item):
    name=item['rfilename'];p=bootstrap.MODEL/name;size=item['size']
    if not p.exists() or p.stat().st_size!=size:
        if size<4*1024*1024:
            for attempt in range(10000):
                try:
                    host='huggingface.co' if attempt%2==0 else 'hf-mirror.com'
                    r=requests.get(f'https://{host}/{META["id"]}/resolve/{META["sha"]}/{name}',timeout=(25,90));r.raise_for_status()
                    assert len(r.content)==size
                    q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(r.content);q.replace(p);break
                except Exception as e:
                    print('RETRY',name,type(e).__name__,flush=True);time.sleep(5)
        else:
            step=256*1024*1024
            ranges=[(start,min(size-1,start+step-1)) for start in range(0,size,step)]
            with ThreadPoolExecutor(max_workers=3) as pool:
                parts=list(pool.map(lambda pair:fetch_range(name,*pair),ranges))
            q=p.with_suffix('.assembling')
            with q.open('wb') as f:
                for part in parts:
                    with part.open('rb') as source:shutil.copyfileobj(source,f,8*1024*1024)
            q.replace(p)
    assert p.stat().st_size==size,name
    expected=item.get('lfs',{}).get('sha256')
    if expected:assert sha(p)==expected, 'SHA256 mismatch: '+name
    with LOCK:STATE['completed_files'].append(name)
    save();print('VERIFIED',name,flush=True)

def main():
    bootstrap.MODEL.mkdir(parents=True,exist_ok=True);save()
    small=[x for x in FILES if x['size']<4*1024*1024]
    large=[x for x in FILES if x not in small]
    for batch in [small,large]:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in as_completed([pool.submit(fetch,x) for x in batch]):result.result()
    STATE.update(status='ready',finished=time.time());save()
    (bootstrap.OUT/'model_integrity.json').write_text(json.dumps({'status':'passed','revision':META['sha'],
        'verification':'Every selected file size and every LFS SHA256 match the pinned Hub manifest'},indent=2)+'\n')

if __name__=='__main__':main()
