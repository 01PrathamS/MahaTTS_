import torch 
from tqdm import tqdm 
import torch.nn.functional as F
import numpy as np
from maha_tts.text.symbols import labels,text_labels,text_labels_en,code_labels,text_enc,text_dec,code_enc,code_dec,text_enc_en,text_dec_en
from scipy.special import softmax 
from maha_tts.utils.audio import denormalize_tacotron_mel,normalize_tacotron_mel,load_wav_to_torch,dynamic_range_compression

def generate_semantic_tokens(
    text,
    model,
    ref_mels,
    language=None,
    temp = 0.7,
    top_p= None,
    top_k= 1, 
    max_tokens = 10, ## previously n_tot_steps
    device = None
    ):
    semb = [] # this may be sementic buffer
    with torch.no_grad():
        for n in tqdm(range(max_tokens)):

            # Tokenizing the text using text_enc or text_enc_en.
            # Adding previously generated tokens (semb) as input codes (semantic tokens).
            # Normalizing the reference mel-spectrogram.
            x = get_inputs(text,semb,ref_mels,device,model.name)


            _,result = model(**x,language=language)  ## result shape is like (1, num_tokens_so_far, vocab_size)

            ## output from the models is 
            # text_probs → shape = [batch, vocab_size, text_length]
            #code_probs → shape = [batch, vocab_size, code_length]

            print("result shape:",result.shape)
            print("-"*20)


            relevant_logits = result[0,:,-1]   ## logits for the last token in the sequence 

            
            print(f"Result: {result.shape}\n, result: {result}")

            print(result[0, :,-1])

            print("relevant_logits shape:",relevant_logits.shape) ## (1, 1004)
            ## i wanted to see what is being generated at each step not entire logits like (1, 1004) just the last dimension from this (1, 1004, x)


            print("*"*20)



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
    text_ids=[text_enc_en['<S>']]+[text_enc_en[i] for i in text.strip()]+[text_enc_en['<E>']] # this is works character by character
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

    from maha_tts.models.autoregressive import TS_model

    #language = torch.tensor([0, 0, 0, 0, 0])  # Assuming all are same language; 0 can mean Marathi for example

    text = "hello how are you my name is pratham savaliya you're awesome thanks for the help" 
    ref_mels = torch.randn(1, 80, 1000)  # Example reference mel spectrogram
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TS_model(n_embed=256, n_layer=6, n_head=4) 
     # Replace with your actual model instance
    # text_ids = torch.randint(0, 100, (5, 20))  ## output 5, 19
    # code_ids = torch.randint(0, 100, (5, 200)) ## output 5, 199
    # speaker_embed = torch.randn((5, 1, 256))

    # 👇 Add this line
    language = torch.tensor([0])  # Assuming all are same language; 0 can mean Marathi for example

    # output = model(
    #     text_ids=text_ids,
    #     speaker_embed=speaker_embed,
    #     codes_ids=code_ids,
    #     language=language,  # 👈 Pass it here!
    #     return_loss=True
    # )

    
    semb,result = generate_semantic_tokens(text, model, ref_mels,language, device=device)
    print("Generated Semantic Tokens:", semb)