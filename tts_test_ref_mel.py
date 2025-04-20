import torch 
from librosa.filters import mel as librosa_mel_fn
from maha_tts.config import config
from maha_tts.utils.audio import dynamic_range_compression
from maha_tts.utils.stft import STFT

from scipy.io.wavfile import read



stft_fn = STFT(config.filter_length, config.hop_length, config.win_length)


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

if __name__ == "__main__":
    import time 

    start_time = time.time()

    ref_clips = []

    import os 
    folder_path = "2272_152282_000019_000001"
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            audio_clip_path = os.path.join(folder_path, file)
            ref_clips.append(audio_clip_path)

    print(get_ref_mels(ref_clips).shape)

    print(f"Time Taken: {time.time() - start_time}")




