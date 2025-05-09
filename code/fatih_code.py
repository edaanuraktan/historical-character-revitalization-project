from TTS.api import TTS
import numpy as np
from scipy.io.wavfile import write
import os

# XTTS modelini yükle
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Ses örneği
speaker_wav_path = "../audio/fatih_sample.wav"

# Metin dosyaları
text_files = [
    "../texts/fatih_text1.txt",
    "../texts/fatih_text2.txt",
    "../texts/fatih_text3.txt",
    "../texts/fatih_text4.txt",
    "../texts/fatih_text5.txt",
    "../texts/fatih_text6.txt",
    "../texts/fatih_text7.txt"
]

# Sesleri tutmak için liste
combined_audio = []

# Sampling rate
sample_rate = 24000  # XTTS modeli genelde bu değeri kullanır

# Metinleri sırayla oku ve sesi oluştur
for file_name in text_files:
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read().strip()

    audio_array = tts.tts(
        text=text,
        speaker_wav=speaker_wav_path,
        language="tr"
    )

    combined_audio.append(audio_array)

# Tüm sesleri birleştir
final_audio = np.concatenate(combined_audio)

# Tek bir .wav dosyası olarak kaydet
write("../audio/fatih_voice.wav", sample_rate, final_audio.astype(np.float32))

print("Birleştirilmiş ses kaydedildi: fatih_voice.wav")