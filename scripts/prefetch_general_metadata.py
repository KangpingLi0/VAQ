import json, subprocess
from pathlib import Path
root=Path('/data/lkp/models/Qwen2.5-VL-7B-Instruct')
items=json.loads(Path('/data/lkp/VAQ/outputs/lat_awq_general_20261107/model_files.json').read_text())
for item in sorted(items,key=lambda x:x.get('size',0)):
 name=item['path']
 if name.startswith('model-0000') and not name.startswith('model-00005'):continue
 if name in ['.gitattributes','README.md']:continue
 target=root/name
 if target.exists():continue
 temporary=root/(name+'.prefetch')
 url='https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/cc594898137f460bfe9f0759e9844b3ce807cfb5/'+name+'?download=true&metadata_prefetch=1'
 subprocess.run(['curl','-sS','--fail','-L','--retry','3','--connect-timeout','20','--max-time','3600','-C','-','-o',str(temporary),url],check=True)
 assert temporary.stat().st_size==item['size'], name
 temporary.replace(target)
 print(name,'ready',flush=True)
