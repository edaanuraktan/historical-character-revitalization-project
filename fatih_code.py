import os
import numpy as np
from scipy.io.wavfile import write
from TTS.api import TTS

# XTTS modeli yükleniyor
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)


# Ses örneği dosyası
speaker_wav_path = "audio/fatih_sample.wav"

# Kullanılacak metin dosyaları
text_files = [
    "text/fatih_text1.txt",
    "text/fatih_text2.txt"
]

# Tüm sesleri birleştirmek için liste
combined_audio = []

# Sampling rate (XTTS için önerilen)
sample_rate = 24000

# Metinleri işleyip sesi oluşturacağız
for file_path in text_files:
    if not os.path.exists(file_path):
        print(f"❌ Dosya bulunamadı: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"🔊 Metin okunuyor ve ses oluşturuluyor: {file_path}")
    audio_array = tts.tts(
        text=text,
        speaker_wav=speaker_wav_path,
        language="tr"
    )

    combined_audio.append(audio_array)

# Sesleri birleştirip kaydetmek için
if combined_audio:
    final_audio = np.concatenate(combined_audio)
    output_path = "audio/fatih_voice2.wav"
    write(output_path, sample_rate, final_audio.astype(np.float32))
    print(f"✅ Birleştirilmiş ses kaydedildi: {output_path}")
else:
    print("❌ Hiçbir ses üretilemedi.")
