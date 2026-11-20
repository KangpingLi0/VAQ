"""Bound eager inference attention memory without truncating tokens or keys."""
import importlib
import math
import torch


def install(full, query_chunk=256):
    for layer in full.language_model.model.layers:
        cls=type(layer.attention)
        if getattr(cls,'_lkp_query_chunked',False):continue
        assert cls.__name__=='InternLM2Attention'
        native=cls.forward
        module=importlib.import_module(cls.__module__)

        def forward(self,hidden_states,attention_mask=None,position_ids=None,
                    past_key_value=None,output_attentions=False,use_cache=False,
                    _native=native,_module=module,**kwargs):
            # Training uses the native attention with layer checkpointing.
            if torch.is_grad_enabled() or output_attentions or hidden_states.shape[1]<=query_chunk:
                return _native(self,hidden_states,attention_mask=attention_mask,position_ids=position_ids,
                    past_key_value=past_key_value,output_attentions=output_attentions,use_cache=use_cache,**kwargs)
            bsz,q_len,_=hidden_states.shape
            qkv=self.wqkv(hidden_states)
            qkv=_module.rearrange(qkv,'b q (h gs d) -> b q h gs d',gs=2+self.num_key_value_groups,d=self.head_dim)
            q=_module.rearrange(qkv[...,:self.num_key_value_groups,:],'b q h gs d -> b q (h gs) d').transpose(1,2)
            k=qkv[...,-2,:].transpose(1,2);v=qkv[...,-1,:].transpose(1,2)
            kv_len=k.shape[-2]+(past_key_value[0].shape[-2] if past_key_value is not None else 0)
            cos,sin=self.rotary_emb(v,seq_len=kv_len)
            q,k=_module.apply_rotary_pos_emb(q,k,cos,sin,position_ids)
            if past_key_value is not None:
                k=torch.cat([past_key_value[0],k],dim=2);v=torch.cat([past_key_value[1],v],dim=2)
            cache=(k,v) if use_cache else None
            k=_module.repeat_kv(k,self.num_key_value_groups)
            v=_module.repeat_kv(v,self.num_key_value_groups)
            chunks=[]
            for start in range(0,q_len,query_chunk):
                stop=min(q_len,start+query_chunk)
                weights=torch.matmul(q[:,:,start:stop],k.transpose(2,3))/math.sqrt(self.head_dim)
                if attention_mask is not None:weights=weights+attention_mask[:,:,start:stop,:]
                weights=torch.nn.functional.softmax(weights,dim=-1,dtype=torch.float32).to(q.dtype)
                chunks.append(torch.matmul(weights,v))
                del weights
            output=torch.cat(chunks,dim=2).transpose(1,2).contiguous().reshape(bsz,q_len,self.hidden_size)
            return self.wo(output),None,cache

        cls.forward=forward
        cls._lkp_query_chunked=True
