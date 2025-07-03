import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ibm-granite/granite-speech-3.3-8b"
speech_granite_processor = AutoProcessor.from_pretrained(
    model_name)
tokenizer = speech_granite_processor.tokenizer
speech_granite = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name).to(device)

# prepare speech and text prompt, using the appropriate prompt template


# Feldolgozandó fájl bekérése
audio_path = input("Add meg a feldolgozandó hangfájl nevét (wav, mp3, ogg): ").strip()
ext = os.path.splitext(audio_path)[1].lower()
if ext not in [".wav", ".mp3", ".ogg"]:
    print("Hiba: csak .wav, .mp3 vagy .ogg fájl dolgozható fel!")
    exit(1)
try:
    wav, sr = torchaudio.load(audio_path, normalize=True)
except Exception as e:
    print(f"Hiba a hangfájl beolvasása közben: {e}")
    exit(1)

audio_duration_seconds = wav.shape[1] / sr
# Ha nem mono, alakítsd át
if wav.shape[0] > 1:
    wav = torch.mean(wav, dim=0, keepdim=True)
# Ha nem 16 kHz, resample
if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    wav = resampler(wav)
    sr = 16000

# Feldarabolás átfedéssel
segment_length = 30  # másodperc
overlap = 5          # másodperc
segments = []
start = 0
while start < wav.shape[1]:
    end = min(start + sr * segment_length, wav.shape[1])
    segments.append(wav[:, start:end])
    if end == wav.shape[1]:
        break
    start += sr * (segment_length - overlap)


# Szegmensek feldolgozása
transcription_start_time = time.time()
all_text = []
srt_entries = []
prev_text = ""
for i, seg in enumerate(segments):
    chat = [
        {
            "role": "system",
            "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
        },
        {
            "role": "user",
            "content": "<|audio|>can you transcribe the speech into a written format?",
        }
    ]
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    model_inputs = speech_granite_processor(
        text,
        seg,
        device=device,
        return_tensors="pt",
    ).to(device)
    model_outputs = speech_granite.generate(
        **model_inputs,
        max_new_tokens=200,
        num_beams=4,
        do_sample=False,
        min_length=1,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)

    output_text = tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True
    )[0].strip()

    # Átfedésből adódó ismétlődés kiszűrése
    if i == 0:
        # Első szegmens: teljes szöveg
        unique_text = output_text
    else:
        # Keresd meg a leghosszabb közös részt az előző szöveg vége és a mostani eleje között
        max_overlap = min(len(prev_text), len(output_text))
        overlap_len = 0
        for j in range(max_overlap, 0, -1):
            if prev_text.endswith(output_text[:j]):
                overlap_len = j
                break
        unique_text = output_text[overlap_len:]

    # SRT időzítés számítása
    seg_start_sec = i * (segment_length - overlap)
    seg_end_sec = seg_start_sec + (seg.shape[1] / sr)
    def sec_to_srt(ts):
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    srt_entry = f"{i+1}\n{sec_to_srt(seg_start_sec)} --> {sec_to_srt(seg_end_sec)}\n{unique_text}\n"
    srt_entries.append(srt_entry)
    print(f"Szegmens {i+1}: {unique_text}")
    all_text.append(unique_text)
    prev_text = output_text




full_transcript = " ".join(all_text)
print("\nTeljes leirat:")
print(full_transcript)

# Leirat mentése txt fájlba
with open("leirat.txt", "w", encoding="utf-8") as f:
    f.write(full_transcript)
transcription_time = time.time() - transcription_start_time
print("\nLeirat elmentve: leirat.txt")

# SRT mentése
with open("leirat.srt", "w", encoding="utf-8") as f:
    f.write("\n".join(srt_entries))
print("SRT elmentve: leirat.srt")


# Írásjel-helyreállítás deepmultilingualpunctuation csomaggal
print("\nÍrásjelezett leirat készítése... (első futásnál letöltés pár perc lehet)")
punctuation_start_time = time.time()
punctuation_time = 0
try:
    from deepmultilingualpunctuation import PunctuationModel
    model = PunctuationModel()
    punctuated = model.restore_punctuation(full_transcript)
    with open("leirat_pontozott.txt", "w", encoding="utf-8") as f:
        f.write(punctuated)
    punctuation_time = time.time() - punctuation_start_time
    print("Írásjelezett leirat elmentve: leirat_pontozott.txt")
except ImportError:
    print("A deepmultilingualpunctuation csomag nincs telepítve. Telepítsd: pip install deepmultilingualpunctuation")
except Exception as e:
    print(f"Hiba történt az írásjelezés során: {e}")

print(f"""
1. A bemeneti hanganyag hossza: {audio_duration_seconds:.2f} másodperc.
2. A leirat.txt elkészítésének ideje: {transcription_time:.2f} másodperc.
3. A leirat_pontozott.txt elkészítésének ideje: {punctuation_time:.2f} másodperc.
""")
