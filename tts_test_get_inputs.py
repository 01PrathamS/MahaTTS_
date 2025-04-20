import torch 

from maha_tts.text.symbols import labels,text_labels,text_labels_en,code_labels,text_enc,text_dec,code_enc,code_dec,text_enc_en,text_dec_en


TACOTRON_MEL_MAX = 2.4
TACOTRON_MEL_MIN = -11.5130


def normalize_tacotron_mel(mel):
    mel = torch.tensor(mel)  # Convert list to tensor
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1



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
  return {'text_ids':torch.tensor(text_ids).unsqueeze(0).to(device),'codes_ids':torch.tensor(semb_ids).unsqueeze(0).to(device),'ref_clips':normalize_tacotron_mel(ref_mels).to(device)}


if __name__ == "__main__":
  text = "hello how are you my name is pratham savaliya you're awesome thanks for the help" 
  res = get_inputs(text)
  print(res["text_ids"], len(res["text_ids"][0]))
  print("-"*100) 
  print(res["codes_ids"])

