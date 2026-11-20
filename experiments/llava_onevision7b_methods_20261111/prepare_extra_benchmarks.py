"""Prepare five pinned additional benchmarks without blocking the core eight."""
import bootstrap
import ast
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import threading
import time

from datasets import Dataset, Features, Image, List, Value, load_dataset
import pandas as pd
from PIL import Image as PILImage
import requests

ROOT=bootstrap.OUT
TASK_ROOT=bootstrap.SOURCE/'3rdparty/lmms-eval-new/lmms_eval/tasks'
LOCK=threading.RLock()
DEFINITIONS=json.loads((ROOT/'extra_benchmarks.json').read_text())['benchmarks']
STATE={'status':'preparing','tasks':{d['task']:{'status':'queued','files_downloaded':0,
    'required_files':len(d['files']),'bytes_total':d['declared_bytes'],'label':d['label']} for d in DEFINITIONS}}


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def update(task,**values):
    with LOCK:
        STATE['tasks'][task].update(values)
        STATE['updated']=time.time()
        save(ROOT/'extra_benchmark_preparation.json',STATE)


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def fetch(definition,item):
    destination=Path(item['target']);destination.parent.mkdir(parents=True,exist_ok=True)
    expected=item.get('lfs',{}).get('sha256')
    if destination.exists():
        assert destination.stat().st_size==item['size']
        if expected:assert sha256(destination)==expected
        return str(destination)
    temporary=destination.with_suffix(destination.suffix+'.download')
    for attempt in range(4):
        offset=temporary.stat().st_size if temporary.exists() else 0
        try:
            while offset<item['size']:
                end=min(item['size']-1,offset+64*1024*1024-1)
                host='huggingface.co' if attempt%3!=2 else 'hf-mirror.com'
                url=f"https://{host}/datasets/{definition['repo']}/resolve/{definition['revision']}/{item['rfilename']}?download=true&offset={offset}&attempt={int(time.time())}"
                with requests.get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(15,45)) as response:
                    response.raise_for_status()
                    if response.status_code==206:
                        assert response.headers['Content-Range']==f"bytes {offset}-{end}/{item['size']}"
                    else:
                        assert response.status_code==200 and offset==0
                    with temporary.open('ab') as handle:
                        for chunk in response.iter_content(256*1024):handle.write(chunk)
                new_offset=temporary.stat().st_size
                assert offset<new_offset<=item['size']
                offset=new_offset
                with LOCK:
                    STATE['tasks'][definition['task']].setdefault('file_bytes',{})[item['rfilename']]=offset
                    save(ROOT/'extra_benchmark_preparation.json',STATE)
            assert temporary.stat().st_size==item['size']
            if expected:assert sha256(temporary)==expected,'Dataset checksum mismatch'
            temporary.replace(destination)
            return str(destination)
        except Exception:
            if attempt==3:raise
            time.sleep(2)


def convert_mmt(path):
    target=path.parent/'mmtbench_val_images.parquet'
    if target.exists():return [str(target)]
    frame=pd.read_csv(path,sep='\t',dtype=object)
    assert not frame['index'].duplicated().any()
    lookup={str(row['index']):row['image'] for _,row in frame.iterrows()}
    def decode(value,seen=()):
        text=str(value).strip()
        if text.startswith('['):
            return [image for item in ast.literal_eval(text) for image in decode(item,seen)]
        if len(text)<64 and text in lookup:
            assert text not in seen,'Cyclic image reference'
            return decode(lookup[text],(*seen,text))
        if text.startswith('data:image/'):text=text.split(',',1)[1]
        content=base64.b64decode(text,validate=True)
        with PILImage.open(io.BytesIO(content)) as image:image.verify()
        return [{'bytes':content,'path':None}]
    columns=list(frame.columns)
    features=Features({name:List(Image()) if name=='image' else Value('string') for name in columns})
    def generate():
        for _,row in frame.iterrows():
            sample={k:None if pd.isna(row[k]) else str(row[k]) for k in columns if k!='image'}
            sample['image']=decode(row['image'])
            assert sample['answer'] in 'ABCDEFGHI' and len(sample['answer'])==1
            yield sample
    # The recursive image-reference resolver is a valid callable, but dill's
    # closure fingerprinting recursively walks itself. Key this conversion by
    # the complete source bytes and converter version instead.
    fingerprint=hashlib.sha256(b'mmt-image-conversion-v1\0'+path.read_bytes()).hexdigest()
    dataset=Dataset.from_generator(generate,features=features,fingerprint=fingerprint)
    temporary=target.with_suffix('.tmp');dataset.to_parquet(str(temporary));temporary.replace(target)
    assert len(dataset)==len(frame)
    return [str(target)]


def local_yaml(definition,files):
    path=TASK_ROOT/definition['yaml']
    backup=path.with_name(path.name+'.upstream')
    if not backup.exists():backup.write_text(path.read_text())
    text=backup.read_text()
    text=re.sub(r'^dataset_path:.*$','dataset_path: parquet',text,flags=re.M)
    text=re.sub(r'^dataset_name:.*\n?','',text,flags=re.M)
    text=re.sub(r'^dataset_kwargs:\n(?:[ \t]+[^\n]*\n?)*','',text,flags=re.M)
    if definition['task']=='naturalbench':
        text=re.sub(r'^doc_to_target:.*$','doc_to_target: "Answer"',text,flags=re.M)
        text=text.replace('metric: naturalbench_ACC','metric: naturalbench_Acc')
    text+='\ndataset_kwargs:\n  data_files:\n    '+definition['split']+':\n'
    text+=''.join('      - '+json.dumps(file)+'\n' for file in files)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(text);temporary.replace(path)


def audit_images(dataset):
    spec=importlib.util.spec_from_file_location('original_image_audit',bootstrap.BASE.parents[1]/'scripts/audit_general_data.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    refs={}
    selected=json.loads((ROOT/'calibration_n128_seed42.json').read_text())
    for row in selected:
        for digest in helper.hashes(Path(row['image']).read_bytes()):refs.setdefault(digest,set()).add(row['id'])
    columns=[k for k,f in dataset.features.items() if isinstance(f,Image)
        or isinstance(f,List) and isinstance(f.feature,Image)
        or k.lower()=='image' and str(f)=="Value('binary')"
        or k=='bytes' and str(f)=="Value('string')"]
    assert columns,'No image feature found'
    count=0;matches=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for batch in dataset.select_columns(columns).data.table.to_batches(max_chunksize=32):
            content=[image for row in batch.to_pylist() for key,v in row.items()
                     for image in ([base64.b64decode(v,validate=True)] if key=='bytes' and isinstance(v,str)
                                   else helper.images(v))]
            for raw,pixels in pool.map(helper.hashes,content):
                count+=1
                ids=refs.get(raw,set())|refs.get(pixels,set())
                if ids:matches.append({'benchmark_image_sha256':raw,'calibration_ids':sorted(ids)})
    return {'image_occurrences':count,'calibration_overlap_occurrences':len(matches),
        'calibration_overlaps':matches,'policy':'Original-byte and decoded-RGB SHA256; full requested test cohort retained, overlaps explicitly flagged rather than silently excluded'}


def prepare(definition):
    task=definition['task'];update(task,status='downloading')
    files=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch,definition,item) for item in definition['files']]):
            files.append(future.result())
            update(task,files_downloaded=len(files))
    files=[item['target'] for item in definition['files']]
    update(task,status='preparing')
    if task=='mmtbench_val':files=convert_mmt(Path(files[0]))
    dataset=load_dataset('parquet',data_files={definition['split']:files},split=definition['split'])
    assert len(dataset)>0
    if task=='mmvp':
        assert len(dataset)==300 and [int(i) for i in dataset['Index']]==list(range(1,301))
    if task=='naturalbench':
        assert len(dataset)==7600 and [int(i) for i in dataset['Index']]==list(range(7600))
        assert 'Answer' in dataset.column_names
    local_yaml(definition,files)
    update(task,status='auditing',rows=len(dataset))
    image_audit=audit_images(dataset)
    prepared=json.loads((ROOT/'benchmark_preparation.json').read_text())
    prepared['tasks'][task]={
        'repo':definition['repo'],'revision':definition['revision'],'split':definition['split'],
        'files':files,'rows':len(dataset),'ready':True,'fingerprint':dataset._fingerprint,
        'primary_metric':definition['primary_metric'],'score_to_percent':definition['score_to_percent'],
        'protocol':definition['protocol'],**image_audit}
    prepared['status']='ready' if len(prepared['tasks'])==13 else 'preparing_extended_benchmarks'
    save(ROOT/'benchmark_preparation.json',prepared)
    update(task,status='ready',rows=len(dataset),**image_audit)
    print('READY',task,len(dataset),'rows; calibration image matches',image_audit['calibration_overlap_occurrences'],flush=True)


def main():
    lock=(ROOT/'extra_benchmarks.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    existing=ROOT/'extra_benchmark_preparation.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        for key,value in old['tasks'].items():STATE['tasks'][key]=value
    order={'mmvp':0,'mmbench_en_dev_static':1,'mmtbench_val':2,'naturalbench':3,'mmerealworld':4}
    while True:
        any_work=False
        for definition in sorted(DEFINITIONS,key=lambda d:order[d['task']]):
            task=definition['task']
            if STATE['tasks'][task]['status']=='ready':continue
            if task!='mmvp' and not(ROOT/'model_integrity.json').exists():
                update(task,status='waiting_for_model_download')
                continue
            any_work=True
            try:prepare(definition)
            except Exception as error:
                import traceback
                traceback.print_exc()
                update(task,status='retrying',error=f'{type(error).__name__}: {error}')
                print('RETRY',task,type(error).__name__,str(error)[:300],flush=True)
        if all(t['status']=='ready' for t in STATE['tasks'].values()):
            STATE['status']='ready';save(existing,STATE);return
        time.sleep(20 if not any_work else 10)


if __name__=='__main__':main()
