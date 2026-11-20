"""Validate the complete CLI and native image path with an explicitly tiny model.

    All synthetic checkpoints and scores stay in integration_only/. They are
    never included in the production experiment plan or result summaries.
"""
import bootstrap
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import torch_npu
from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration, Qwen2Config, SiglipVisionConfig


def main():
    torch.set_num_threads(4)
    root=bootstrap.OUT/'integration_only'
    checkpoint=root/'synthetic_checkpoint'
    output=root/'synthetic_results'
    checkpoint.mkdir(parents=True,exist_ok=True)
    output.mkdir(exist_ok=True)
    (output/'logs').mkdir(exist_ok=True)
    original=json.loads((bootstrap.MODEL/'config.json').read_text())
    text=Qwen2Config(vocab_size=152128,hidden_size=128,intermediate_size=256,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
        bos_token_id=151643,eos_token_id=151645,pad_token_id=151643)
    vision=SiglipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,image_size=384,patch_size=14)
    config=LlavaOnevisionConfig(text_config=text,vision_config=vision,
        image_token_index=151646,video_token_index=151647,
        image_grid_pinpoints=original['image_grid_pinpoints'])
    config._attn_implementation='sdpa'
    torch.manual_seed(42)
    tiny=LlavaOnevisionForConditionalGeneration(config).half().eval()
    tiny.save_pretrained(str(checkpoint))
    del tiny
    gc.collect()
    for file in bootstrap.MODEL.rglob('*'):
        if not file.is_file() or file.suffix not in ('.json','.txt'):continue
        if file.name in ('config.json','generation_config.json') or file.name.startswith('model.'):continue
        destination=checkpoint/file.relative_to(bootstrap.MODEL)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,destination)
    shutil.copy2(bootstrap.OUT/'calibration_n128_seed42.json',output/'calibration_n128_seed42.json')
    env=dict(os.environ,LKP_ONEVISION_MODEL_DIR=str(checkpoint),LKP_ONEVISION_OUTPUT=str(output))
    def run(label,args):
        with (output/'logs'/f'{label}.log').open('w') as log:
            subprocess.run([sys.executable,'-u',*args],env=env,cwd=bootstrap.SOURCE,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print('PASSED synthetic integration',label,flush=True)
    run('calibration',[str(bootstrap.BASE/'run_model.py'),'calibration','--n','2'])
    run('rtn_w4a8',[str(bootstrap.BASE/'run_model.py'),'quant','--method','rtn','--w','4','--a','8','--n','2'])
    for variant,state in [('fp16',None),('rtn_w4a8',output/'rtn_w4a8_smoke/quantized_state.pt')]:
        model_args=f'pretrained={checkpoint},torch_dtype=float16,attn_implementation=sdpa'
        if state:model_args+=f',state_path={state},w_bit=4,a_bit=8'
        destination=output/'eval'/variant
        run('evaluate_'+variant,[str(bootstrap.BASE/'evaluate_model.py'),'--model','llava_onevision',
            '--model_args',model_args,'--tasks','ai2d','--device','npu','--batch_size','1','--seed','42',
            '--gen_kwargs','temperature=0,do_sample=False,max_new_tokens=8','--bootstrap_iters','0',
            '--log_samples','--limit','2','--output_path',str(destination),'--verbosity','DEBUG'])
        results=list(destination.rglob('*_results.json'))
        assert results
        data=json.loads(results[-1].read_text())
        assert data['n-samples']['ai2d']['effective']==2
    (bootstrap.OUT/'integration_validation.json').write_text(json.dumps({'status':'passed','finished':time.time(),
        'scope':'Synthetic tiny checkpoint with the real OneVision processor/vocabulary and native 384-pixel image path; no model-quality claim',
        'checks':['Actual multimodal embedding capture and assistant labels','Real RTN W4A8 path',
            'Strict full-state reload','Unchanged evaluation CLI, datasets, generation, scoring and sample serialization'],
        'artifacts':str(root),'production_results_modified':False},indent=2)+'\n')


if __name__=='__main__':main()
