"""Audit all four calibration seeds against the supplemental full cohorts."""
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import io
import json
import os
from pathlib import Path
os.environ.setdefault('HF_DATASETS_CACHE','/data/lkp/datasets/llava_onevision_hf_cache')
from datasets import load_dataset, Image, List
from PIL import Image as PILImage
import schedule

def hashes(raw):
    with PILImage.open(io.BytesIO(raw)) as image:
        image=image.convert('RGB');pixels=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    return hashlib.sha256(raw).hexdigest(),pixels

def images(value):
    if isinstance(value,bytes):yield value
    elif isinstance(value,dict):
        if value.get('bytes') is not None:yield value['bytes']
        elif value.get('path'):yield Path(value['path']).read_bytes()
    elif isinstance(value,list):
        for item in value:yield from images(item)

def main():
    os.nice(10);refs={}
    for seed in [42,123,2026,3407]:
        directory='lat_awq_general_seed3407_20261111' if seed==3407 else 'lat_awq_general_20261107'
        rows=json.loads((schedule.ROOT/'outputs'/directory/f'calibration_n128_seed{seed}.json').read_text())
        for row in rows:
            for digest in hashes(Path(row['image']).read_bytes()):refs.setdefault(digest,set()).add((seed,row['id']))
    result={'status':'auditing','seeds':[42,123,2026,3407],'tasks':{},'policy':'Full cohorts retained; exact byte/RGB overlaps explicitly recorded'}
    path=schedule.OUT/'audit_extra_all_seeds.json';schedule.save(path,result)
    for task in [d['task'] for d in schedule.read(schedule.OUT/'extra_benchmarks.json')['benchmarks']]:
        try:
            definition=schedule.PREP[task]
            ds=load_dataset('parquet',data_files={definition['split']:definition['files']},split=definition['split'])
            columns=[k for k,f in ds.features.items() if isinstance(f,Image) or isinstance(f,List) and isinstance(f.feature,Image)
                or k.lower()=='image' and str(f)=="Value('binary')" or k=='bytes' and str(f)=="Value('string')"]
            assert columns
            count=0;overlaps=[]
            with ThreadPoolExecutor(max_workers=4) as pool:
                for batch in ds.select_columns(columns).data.table.to_batches(max_chunksize=16):
                    content=[raw for row in batch.to_pylist() for key,value in row.items()
                        for raw in ([base64.b64decode(value,validate=True)] if key=='bytes' and isinstance(value,str) else images(value))]
                    for raw,pixel in pool.map(hashes,content):
                        count+=1;matches=refs.get(raw,set())|refs.get(pixel,set())
                        if matches:overlaps.append({'image_sha256':raw,'matches':[{'seed':seed,'id':id} for seed,id in sorted(matches)]})
            result['tasks'][task]={'status':'complete','image_occurrences':count,'overlap_occurrences':len(overlaps),'overlaps':overlaps}
        except Exception as error:result['tasks'][task]={'status':'failed','error':repr(error)}
        schedule.save(path,result)
    result['status']='complete' if all(v['status']=='complete' for v in result['tasks'].values()) else 'errors';schedule.save(path,result)

if __name__=='__main__':main()
