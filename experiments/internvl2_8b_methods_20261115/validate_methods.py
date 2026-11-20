"""Exercise all eleven real quantization paths on a small InternVL2 decoder."""
import bootstrap
import copy
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from model_bridge import QuantizationAdapter, decoder_view
from test_bridge import tiny_model


def main():
    from qmllm.quantization.quant_wrapper import qwrapper
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    root = bootstrap.OUT/'method_validation'
    root.mkdir(exist_ok=True)
    methods = [(3,16,m) for m in ['rtn','awq','gptq','mbq','qig','lat_awq']]
    methods += [(4,8,m) for m in ['rtn','smoothquant','mbq','qig','lat_awq']]
    checks = []
    for w,a,method in methods:
        torch.manual_seed(42)
        full = tiny_model().half().to('npu:0')
        adapter = QuantizationAdapter(full)
        original = copy.deepcopy(adapter.model.state_dict())
        before = adapter.model.model.layers[0].attention.wqkv.weight.detach().clone()
        vision = torch.zeros(2,8,dtype=torch.bool)
        vision[:,:3] = True
        labels = torch.randint(1,200,(2,8))
        labels[:,:5] = -100
        inputs = {'inputs_embeds': torch.randn(2,8,128,dtype=torch.float16)}
        kwargs = {'attention_mask':torch.ones(2,8,dtype=torch.long),'labels':labels,
                  'vision_mask':vision,'caption_mask':labels!=-100}
        run = root/f'{method}_w{w}a{a}'
        run.mkdir(exist_ok=True)
        options = SimpleNamespace(method=method,model='internvl2',model_args='',
            w_bit=w,a_bit=a,w_group=128,alpha=.5,run_process=True,pseudo_quant=True,
            reweight=method in ('mbq','qig'),distort=False,loss_mode='mae',
            scale_path=str(run/'scales.pt'),torch_device=torch.device('npu:0'),percdamp=.01,
            token_aware_saliency=method=='lat_awq',token_weighted_loss=method=='lat_awq',
            saliency_mix_lambda=.5,lat_debug=False,lat_log_dir=str(run),lat_output_dir=str(run),
            effective_n_samples=2,seed=42)
        if method in ('awq','mbq','qig','lat_awq'):
            options.pseudo_quant = False
            qwrapper(adapter,inputs,kwargs,options)
            adapter.to_cpu()
            adapter.model.load_state_dict(original,strict=True)
            options.run_process = False
            options.pseudo_quant = True
            qwrapper(adapter,None,None,options)
        else:
            qwrapper(adapter,inputs,kwargs,options)
        with torch.no_grad():
            result=adapter.model(input_ids=torch.tensor([[1,2,3,4]],device='npu:0'))
        assert torch.isfinite(result.logits).all()
        after=adapter.model.model.layers[0].attention.wqkv.weight
        assert not torch.equal(before.to(after.device),after)
        assert adapter.model is full.language_model
        checks.append({'method':method,'w':w,'a':a,'finite_logits':True,'weight_changed':True})
        print('PASSED',method,f'W{w}A{a}',flush=True)
        del full,adapter,original,before,result,inputs,kwargs
        gc.collect()
        torch.npu.empty_cache()
    (bootstrap.OUT/'method_validation.json').write_text(json.dumps({'status':'passed','finished':time.time(),
        'checks':checks,'scope':'Small real InternVL2/InternLM2 decoder; all requested quantizers, before full-checkpoint smoke tests'},indent=2)+'\n')


if __name__=='__main__':
    main()
