"""Official dynamic image preprocessing and native chat prompt/embeddings."""
import bootstrap
import importlib
import re
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    # Same ratio ordering and tie rule as the pinned official model-card code.
    width,height=image.size;aspect=width/height
    ratios=set((i,j) for n in range(min_num,max_num+1)
        for i in range(1,n+1) for j in range(1,n+1) if min_num<=i*j<=max_num)
    ratios=sorted(ratios,key=lambda pair:pair[0]*pair[1])
    best=(1,1);diff=float('inf')
    for ratio in ratios:
        candidate=abs(aspect-ratio[0]/ratio[1])
        if candidate<diff or candidate==diff and width*height>0.5*image_size**2*ratio[0]*ratio[1]:
            best=ratio;diff=candidate
    resized=image.resize((image_size*best[0],image_size*best[1]))
    tiles=[resized.crop(((i%best[0])*image_size,(i//best[0])*image_size,
        (i%best[0]+1)*image_size,(i//best[0]+1)*image_size)) for i in range(best[0]*best[1])]
    if use_thumbnail and len(tiles)!=1:tiles.append(image.resize((image_size,image_size)))
    return tiles


def conversation(full):
    module=importlib.import_module(full.__class__.__module__.rsplit('.',1)[0]+'.conversation')
    conv=module.get_conv_template(full.template)
    conv.system_message=full.system_message
    return conv


def prepare(full, tokenizer, context, images, answer=None, device=None):
    device=device or next(full.parameters()).device
    size=full.config.force_image_size or full.config.vision_config.image_size
    transform=T.Compose([T.Lambda(lambda image:image.convert('RGB')),
        T.Resize((size,size),interpolation=InterpolationMode.BICUBIC),T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])
    tiles=[];counts=[]
    for image in images:
        crops=dynamic_preprocess(image.convert('RGB'),min_num=full.config.min_dynamic_patch,
            max_num=full.config.max_dynamic_patch,image_size=size,use_thumbnail=True)
        counts.append(len(crops));tiles.extend(transform(crop) for crop in crops)
    context=re.sub(r'<image(?:\s+\d+)?>','',context).strip()
    if len(images)==1:context='<image>\n'+context
    elif images:context=''.join(f'Image-{i+1}: <image>\n' for i in range(len(images)))+context
    conv=conversation(full)
    conv.append_message(conv.roles[0],context);conv.append_message(conv.roles[1],None)
    prompt=conv.get_prompt()
    for count in counts:
        prompt=prompt.replace('<image>','<img>'+'<IMG_CONTEXT>'*(full.num_image_token*count)+'</img>',1)
    prefix=tokenizer(prompt,return_tensors='pt')
    complete=prompt if answer is None else prompt+answer+conv.sep
    inputs=tokenizer(complete,return_tensors='pt')
    assert torch.equal(inputs['input_ids'][:,:prefix['input_ids'].shape[1]],prefix['input_ids']), 'Assistant boundary changed'
    ids=inputs['input_ids'].to(device);mask=inputs['attention_mask'].to(device)
    embeds=full.language_model.get_input_embeddings()(ids).clone()
    selected=ids==full.img_context_token_id
    pixels=None
    if tiles:
        pixels=torch.stack(tiles).to(device=device,dtype=next(full.parameters()).dtype)
        features=full.extract_feature(pixels).reshape(-1,embeds.shape[-1])
        assert int(selected.sum())==features.shape[0], 'Image embedding/token count mismatch'
        embeds[selected]=features
    else:assert not selected.any()
    labels=ids.clone();labels[:,:prefix['input_ids'].shape[1]]=-100
    assert not ((labels!=-100)&selected).any()
    return {'inputs_embeds':embeds,'input_ids':ids,'attention_mask':mask,'labels':labels,
            'vision_mask':selected,'caption_mask':labels!=-100,'eos_token_id':tokenizer.convert_tokens_to_ids(conv.sep.strip()),
            'num_patches':counts,'pixel_values':pixels}


@torch.inference_mode()
def greedy_generate(full, inputs, max_new_tokens=256):
    """Native legacy KV tuples; equivalent batch-one greedy next-token decoding.

    The pinned 2024 remote model uses tuple KV caches. Current Transformers
    generation assumes newer Cache objects. Calling the unchanged native
    forward keeps that model's cache protocol and attention equations intact.
    """
    lm=full.language_model
    mask=inputs['attention_mask'];past=None;next_ids=None;generated=[]
    # Only last-token logits are used in greedy generation. This avoids a full
    # prompt-length x 92544 logits allocation without altering selected logits.
    hook=lm.get_output_embeddings().register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
    try:
        for step in range(max_new_tokens):
            kwargs={'inputs_embeds':inputs['inputs_embeds']} if step==0 else {'input_ids':next_ids}
            output=lm(**kwargs,attention_mask=mask,past_key_values=past,use_cache=True,return_dict=True)
            logits=output.logits[:,-1,:]
            if not torch.isfinite(logits).all():raise FloatingPointError('Non-finite generation logits')
            next_ids=logits.argmax(-1,keepdim=True)
            generated.append(next_ids)
            if int(next_ids.item())==inputs['eos_token_id']:break
            past=output.past_key_values
            mask=torch.cat([mask,torch.ones_like(next_ids)],dim=1)
    finally:hook.remove()
    return torch.cat(generated,dim=1)
