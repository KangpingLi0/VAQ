"""Finish partial weight shards using verified HTTP byte ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import psutil
import requests

import bootstrap


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    info=json.loads((bootstrap.OUT/'model_info.json').read_text())
    shards=[x for x in info['siblings'] if x['rfilename'].endswith('.safetensors')]
    pending=[]
    snapshots={}
    for item in shards:
        target=bootstrap.MODEL/item['rfilename']
        if target.exists():continue
        partial=target.with_name(target.name+'.download')
        prefix=target.with_name(target.name+'.range_prefix')
        if prefix.exists():
            length=prefix.stat().st_size
        else:
            length=partial.stat().st_size if partial.exists() else 0
            with prefix.open('wb') as dst:
                if length:
                    with partial.open('rb') as src:
                        remaining=length
                        while remaining:
                            chunk=src.read(min(16*1024*1024,remaining))
                            if not chunk:raise IOError('Incomplete prefix snapshot')
                            dst.write(chunk);remaining-=len(chunk)
        snapshots[target]=(item,prefix,length)
        rest=item['size']-length
        for i in range(3):
            start=length+rest*i//3
            end=length+rest*(i+1)//3-1
            if start<=end:pending.append((item,start,end))

    def fetch(job):
        item,start,end=job
        part=bootstrap.MODEL/f"{item['rfilename']}.range_{start}_{end}"
        if part.exists() and part.stat().st_size==end-start+1:return part
        for attempt in range(12):
            existing=part.stat().st_size if part.exists() else 0
            assert existing<=end-start+1
            offset=start+existing
            if offset>end:return part
            host='huggingface.co' if attempt%3!=2 else 'hf-mirror.com'
            url=f"https://{host}/{info['id']}/resolve/{info['sha']}/{item['rfilename']}?download=true&offset={offset}&attempt={int(time.time())}"
            try:
                with requests.get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(20,60)) as response:
                    response.raise_for_status()
                    assert response.status_code==206
                    assert response.headers['Content-Range']==f"bytes {offset}-{end}/{item['size']}"
                    with part.open('ab') as handle:
                        for chunk in response.iter_content(256*1024):handle.write(chunk)
                assert part.stat().st_size==end-start+1
                print('Completed range',item['rfilename'],start,end,flush=True)
                return part
            except Exception as error:
                print('Resuming range',item['rfilename'],start,'saved bytes',part.stat().st_size if part.exists() else 0,type(error).__name__,flush=True)
                if attempt==11:raise
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=9) as pool:
        for future in as_completed([pool.submit(fetch,job) for job in pending]):future.result()
    for target,(item,prefix,length) in snapshots.items():
        temporary=target.with_name(target.name+'.range_complete')
        parts=[prefix]+[bootstrap.MODEL/f"{item['rfilename']}.range_{start}_{end}"
            for f,start,end in pending if f['rfilename']==item['rfilename']]
        with temporary.open('wb') as out:
            for part in parts:
                with part.open('rb') as inp:
                    for chunk in iter(lambda:inp.read(16*1024*1024),b''):out.write(chunk)
        assert temporary.stat().st_size==item['size']
        assert digest(temporary)==item['lfs']['sha256'],'Assembled checkpoint checksum mismatch'
        temporary.replace(target)
        print('Verified assembled shard',target.name,flush=True)
    # Stop only the redundant original downloader and its curl children.
    original_pid=int((bootstrap.OUT/'original_download.pid').read_text())
    try:
        process=psutil.Process(original_pid)
        if 'download_model.py' in ' '.join(process.cmdline()):
            children=process.children(recursive=True)
            process.terminate()
            for child in children:
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            psutil.wait_procs([process,*children],timeout=5)
    except psutil.NoSuchProcess:
        pass
    files=[f for f in info['siblings'] if f['rfilename'].endswith(('.json','.txt','.safetensors'))]
    for item in files:
        path=bootstrap.MODEL/item['rfilename']
        assert path.stat().st_size==item['size']
        if item.get('lfs'):assert digest(path)==item['lfs']['sha256']
    status={'status':'ready','repository':info['id'],'revision':info['sha'],'required_files':len(files),
        'completed_files':[f['rfilename'] for f in files],'model_dir':str(bootstrap.MODEL),'finished':time.time()}
    for name,data in [('download_status.json',status),('model_integrity.json',{'status':'passed',
        'revision':info['sha'],'verification':'Every file size checked and all four weight SHA256 hashes matched the pinned Hub manifest'})]:
        path=bootstrap.OUT/name;tmp=path.with_suffix('.range_tmp')
        tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
    for target,(item,prefix,length) in snapshots.items():
        prefix.unlink(missing_ok=True)
        target.with_name(target.name+'.download').unlink(missing_ok=True)
    for item,start,end in pending:
        (bootstrap.MODEL/f"{item['rfilename']}.range_{start}_{end}").unlink(missing_ok=True)


if __name__=='__main__':main()
