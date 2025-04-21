'''
Inspiration taken from https://github.com/neonbjb/tortoise-tts/blob/main/tortoise/models/autoregressive.py
'''


## this codd is about converting text to semantic not text to speech.

## text => M1 => semantic => M2 (Diffusion) ## => mel spectrogram => vocoder => audio

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import functools

from typing import Any
from torch.utils.data import Dataset,DataLoader

## let's change this to vllm implementation
from transformers import GPT2Config, GPT2Model
#from vllm.vllm.model_executor.models.gpt2 import GPT2Model


from tqdm import tqdm
from maha_tts.config import config
from maha_tts.text.symbols import labels,code_labels,text_labels,text_labels_en
from maha_tts.models.modules import GST

def null_position_embeddings(range, dim): ## returns (batch, row, column)
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

class TS_model(nn.Module):
    def __init__(self,n_embed = 512, n_layer = 16, n_head = 8, n_positions = 2048, name='Smolie-in'):
        super(TS_model,self).__init__()

        self.vocab_size=len(labels)  ## 10682 
        self.n_positions=n_positions  ## 2048
        self.n_embed=n_embed  ## 512 
        self.n_layer=n_layer ## 16
        self.n_head=n_head ## 8
        self.name=name  ## Smolie-en

        self.config = GPT2Config(vocab_size=self.vocab_size,n_positions=self.n_positions,n_embd=self.n_embed,n_layer=self.n_layer,n_head=self.n_head)
        # self.gpt = GPT2Model(self.config)

        self.gpt = GPT2Model(self.config) ## vllm implementation


        


        del self.gpt.wpe
        self.gpt.wpe = functools.partial(null_position_embeddings, dim=self.n_embed) ## null position embeddings no upyog aaya thayelo che.
        # Built-in token embeddings are unused.
        del self.gpt.wte
        self.GST = GST(model_channels=self.n_embed,num_heads=self.n_head,in_channels=config.n_mel_channels,k=1) ## input: 512, 8, 80, 1
        ## gst output : this returns architecutre 

        #print(len(text_labels_en), len(text_labels))  ## (52, 679)
        if self.name == 'Smolie-en':
            self.text_head = nn.Linear(self.n_embed,len(text_labels_en)) # n_embed=512, 
        else:
            self.text_head = nn.Linear(self.n_embed,len(text_labels))

        #print(len(code_labels)) ## 1004

        self.code_head = nn.Linear(self.n_embed,len(code_labels))
        self.text_positional_embed = LearnedPositionEmbeddings(self.n_positions,self.n_embed)
        self.code_positional_embed = LearnedPositionEmbeddings(self.n_positions,self.n_embed)
        
        if self.name == 'Smolie-en':
            self.text_embed = nn.Embedding(len(text_labels_en),self.n_embed) ## nn.Embedding(52, 2048)
        else:
            self.text_embed = nn.Embedding(len(text_labels),self.n_embed)
        self.code_embed = nn.Embedding(len(code_labels),self.n_embed)  ## nn.Embed(1004, 2048)
        if self.name != 'Smolie-en':
            self.language_embed = nn.Embedding(len(config.lang_index),self.n_embed)  ## nn.Embedding(9, 2048)

        self.final_norm = nn.LayerNorm(self.n_embed)

    def get_speaker_latent(self, ref_mels):
        ref_mels = ref_mels.unsqueeze(1) if len(
            ref_mels.shape) == 3 else ref_mels

        conds = []
        for j in range(ref_mels.shape[1]):
            conds.append(self.GST(ref_mels[:, j,:,:]))

        conds = torch.cat(conds, dim=-1)
        conds = conds.mean(dim=-1)

        return conds.unsqueeze(1)

    def forward(self,text_ids,codes_ids = None,speaker_embed=None,ref_clips=None,language=None,return_loss = False):
        assert speaker_embed is not None or ref_clips is not None
        text_embed = self.text_embed(text_ids)
        text_embed += self.text_positional_embed(text_embed)
        if self.name != 'Smolie-en':
            text_embed += self.language_embed(language).unsqueeze(1)
        code_embed = None
        code_probs= None

        if codes_ids is not None:
            code_embed = self.code_embed(codes_ids)
            code_embed+= self.code_positional_embed(code_embed)

        if ref_clips is not None:
            speaker_embed = self.get_speaker_latent(ref_clips)

        text_embed,code_embed = self.get_logits(speaker_embed=speaker_embed,text_embed=text_embed,code_embed=code_embed)

        text_probs = self.text_head(text_embed).permute(0,2,1)
        
        if codes_ids is not None:
            code_probs = self.code_head(code_embed).permute(0,2,1)

        if return_loss:
            loss_text = F.cross_entropy(text_probs[:,:,:-1], text_ids[:,1:].long(), reduce=False)
            loss_mel = F.cross_entropy(code_probs[:,:,:-1], codes_ids[:,1:].long(), reduce=False)
            return loss_text,loss_mel,code_probs
        
        return text_probs,code_probs


    def get_logits(self,speaker_embed,text_embed,code_embed=None):
        
        if code_embed is not None:
            embed = torch.cat([speaker_embed,text_embed,code_embed],dim=1)
        else:
            embed = torch.cat([speaker_embed,text_embed],dim=1)
        
        #gpt_output = self.gpt(inputs_embeds=embed, return_dict=True) # pass through GPT-2 model and returns contextual embeddings for each input token
        #enc = gpt_output.last_hidden_state[:, 1:] ## this removes speaker token embedding

## let's change this from standard transformers gpt2 implementation to vllm gpt2 implementation

        gpt_output = self.gpt(inputs_embeds=embed) # pass through GPT-2 model and returns contextual embeddings for each input token
        enc = gpt_output[:, 1:] # kem k vllm na implementation ma direct hidden_states aave che.

        enc = self.final_norm(enc)

        #  split the output into the text and code parts

        if code_embed is not None:
            return enc[:,:text_embed.shape[1]],enc[:,-code_embed.shape[1]:] # Return the contextual embeddings for the text tokens and code tokens separately
        
        return enc[:,:text_embed.shape[1]],None

## this one is just simple thing
class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)

def load_TS_model(checkpoint,device,name):
    data = torch.load(checkpoint,map_location=torch.device('cpu'))
    sem_model= TS_model(n_embed = data['n_embed'], n_layer = data['n_layer'], n_head = data['n_head'], n_positions = data['n_positions'],name=name)
    print(name,data['n_embed'],data['n_layer'],data['n_head'],data['n_positions'])
    sem_model.load_state_dict(data['state_dict'],strict=True)
    sem_model.eval().to(device)

    return sem_model

#if __name__ == '__main__':
    # print("hello world")
    # model=TS_model(n_embed = 256, n_layer = 6, n_head = 4)

    # import time 

    # start_time = time.time()

    # text_ids = torch.randint(0,100,(5,20))
    # code_ids = torch.randint(0,100,(5,200))
    # speaker_embed = torch.randn((5,1,256))

    # output=model(text_ids=text_ids,speaker_embed=speaker_embed,codes_ids=code_ids,return_loss=True)

    # print(f"Time: {time.time()-start_time} seconds")

if __name__ == '__main__':
    import time 
    start_time = time.time()
    print("hello world")
    model = TS_model(n_embed=256, n_layer=6, n_head=4)
    


    text_ids = torch.randint(0, 100, (5, 20))  ## output 5, 19
    code_ids = torch.randint(0, 100, (5, 200)) ## output 5, 199
    speaker_embed = torch.randn((5, 1, 256))

    # 👇 Add this line
    language = torch.tensor([0, 0, 0, 0, 0])  # Assuming all are same language; 0 can mean Marathi for example

    output = model(
        text_ids=text_ids,
        speaker_embed=speaker_embed,
        codes_ids=code_ids,
        language=language,  # 👈 Pass it here!
        return_loss=True
    )

    print(output)
    print(type(output))
    print(output[0].shape) ## returns (ref_clips, text_tokens)

    print(f"Time: {time.time() - start_time} seconds")
