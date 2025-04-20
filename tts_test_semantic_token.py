

import torch 
from tqdm import tqdm 
import numpy as np 
from scipy.special import softmax

import torch.nn.functional as F
from maha_tts.text.symbols import labels,text_labels,text_labels_en,code_labels,text_enc,text_dec,code_enc,code_dec,text_enc_en,text_dec_en

import torch 
from librosa.filters import mel as librosa_mel_fn
from maha_tts.config import config
from maha_tts.utils.audio import dynamic_range_compression
from maha_tts.utils.stft import STFT

from scipy.io.wavfile import read



stft_fn = STFT(config.filter_length, config.hop_length, config.win_length)


TACOTRON_MEL_MAX = 2.4
TACOTRON_MEL_MIN = -11.5130


def denormalize_tacotron_mel(norm_mel):
    return ((norm_mel+1)/2)*(TACOTRON_MEL_MAX-TACOTRON_MEL_MIN)+TACOTRON_MEL_MIN


def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1




def get_ref_mels(ref_clips):
    ref_mels = []
    for i in ref_clips:
        ref_mels.append(get_mel(i)[0][:,:500])
    
    ref_mels_padded = (torch.randn((len(ref_mels), 80, 500)))*1e-8
    for i,mel in enumerate(ref_mels):
        ref_mels_padded[i, :, :mel.size(1)] = mel
    return ref_mels_padded.unsqueeze(0)



def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path,)
    return torch.FloatTensor(data), sampling_rate


mel_basis = librosa_mel_fn(
        sr=config.sampling_rate, n_fft=config.filter_length, n_mels=config.n_mel_channels, fmin=config.mel_fmin, fmax=config.mel_fmax)


def get_mel(filepath):
    audio, sampling_rate = load_wav_to_torch(filepath)
    #audio_norm = audio / config.MAX_WAV_VALUE
    audio_norm = audio / 32768.0
    audio_norm = audio_norm.unsqueeze(0)
    y = torch.autograd.Variable(audio_norm, requires_grad=False)
    assert(torch.min(y.data) >= -1)
    assert(torch.max(y.data) <= 1)
    magnitudes, phases = stft_fn.transform(y)
    magnitudes = magnitudes.data
    assert magnitudes.shape == torch.tensor(magnitudes).shape
    magnitudes = torch.tensor(magnitudes)
    mel_output = torch.matmul(torch.tensor(mel_basis), magnitudes)
    mel_output = dynamic_range_compression(mel_output)
    melspec = torch.squeeze(mel_output, 0)
    energy = torch.norm(magnitudes, dim=1).squeeze(0)
    return melspec,list(energy)


def generate_semantic_tokens(
    text,
    model,
    ref_mels,
    language=None,
    temp = 0.7,
    top_p= None,
    top_k= 1,
    n_tot_steps = 100,
    device = None
    ):
    semb = [] # this may be sementic buffer
    with torch.no_grad():
        for n in tqdm(range(n_tot_steps)):
            x = get_inputs(text,semb,ref_mels,device,model.name)
            _,result = model(**x,language=language)
            relevant_logits = result[0,:,-1]
            if top_p is not None:
                # faster to convert to numpy
                original_device = relevant_logits.device
                relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
                relevant_logits = relevant_logits.to(original_device)

            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")

            probs = F.softmax(relevant_logits / temp, dim=-1)
            item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
            semb.append(str(code_dec[item_next.item()]))
            if semb[-1] == '<EST>' or semb[-1] == '<PAD>':
                break

            del relevant_logits, probs, item_next

    semb = torch.tensor([int(i) for i in semb[:-1]])
    return semb,result

def get_inputs(text,semb=[],ref_mels=[],device=torch.device('cpu'),name = 'Smolie-in'):
  text = text.lower()
  if name=='Smolie-en':
    text_ids=[text_enc_en['<S>']]+[text_enc_en[i] for i in text.strip()]+[text_enc_en['<E>']]
  else:
    text_ids=[text_enc['<S>']]+[text_enc[i] for i in text.strip()]+[text_enc['<E>']]
    
  semb_ids=[code_enc['<SST>']]+[code_enc[i] for i in semb]#+[tok_enc['<EST>']]

  input_ids = text_ids+semb_ids
  # pad_length = config.t2s_position-(len(text_ids)+len(semb_ids))

  token_type_ids = [0]*len(text_ids)+[1]*len(semb_ids)#+[0]*pad_length
  positional_ids = [i for i in range(len(text_ids))]+[i for i in range(len(semb_ids))]#+[0]*pad_length
  # labels = [-100]*len(text_ids)+semb_ids+[-100]*pad_length
  attention_mask = [1]*len(input_ids)#+[0]*pad_length
  # input_ids += [tok_enc['<PAD>']]*pad_length

  print("-"*100) 
  print(f"Text: {text_ids}, Length: {len(text_ids)}, Semantic: {semb_ids}, length: {len(semb_ids)}")

 
  return {'text_ids':torch.tensor(text_ids).unsqueeze(0).to(device),'codes_ids':torch.tensor(semb_ids).unsqueeze(0).to(device),'ref_clips':normalize_tacotron_mel(ref_mels).to(device)}


if __name__ == "__main__":

    import time 


    start_time = time.time()
   
    ref_clips = []
    from maha_tts.models.autoregressive import TS_model

    import os 
    folder_path = "2272_152282_000019_000001"
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            audio_clip_path = os.path.join(folder_path, file)
            ref_clips.append(audio_clip_path)
    model = TS_model(n_embed=256, n_layer=6, n_head=4)

    text = "what you are upto in united states of "

    language = torch.tensor([0])



    res = generate_semantic_tokens(text, model, get_ref_mels(ref_clips), language=language)

    print(res[0].shape, res[1].shape)


    print(res[0])

    #logits_step0 = res[1][0, 0, :]  # shape: [10004]
    #print(logits_step0)

    print(res[0].max(), res[0].min())

    print(f"Time Taken: {time.time() - start_time}")
