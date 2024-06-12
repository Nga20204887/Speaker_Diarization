import os
import re
import time

import librosa
import torch
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import pandas as pd
from flask import Flask, request, jsonify
import librosa
import numpy as np
import re
import os

def extract_speaker_time_intervals(rttm_file):
    with open(rttm_file, 'r') as file:
        lines = file.readlines()

    # Define a regular expression pattern to match lines with speaker information
    pattern = re.compile(r'^SPEAKER\s+(\S+)\s+1\s+(\S+)\s+(\S+)\s+<NA>\s+<NA>\s+(\S+)\s+<NA>\s+<NA>\s*$')

    # Extract time intervals and corresponding speaker IDs
    speaker_intervals = []
    for line in lines:
        match = pattern.match(line)
        if match:
            speaker_id = match.group(4)
            start_time = float(match.group(2))
            duration = float(match.group(3))
            end_time = start_time + duration

            speaker_intervals.append((start_time, end_time, speaker_id))

    # Sort intervals based on start time
    speaker_intervals.sort(key=lambda x: x[0])

    return speaker_intervals

def cut_and_save_audio(input_audio_path,  intervals, sr=16000):
    audio, sr = librosa.load(input_audio_path, sr=sr)
    text =""
    for i, (start, end, speaker_id) in enumerate(intervals, 1):
        # Convert start and end times to sample indices
        start_idx = int(start * sr)
        end_idx = int(end * sr)

        # Extract the segment
        segment = audio[start_idx:end_idx]

        text +=  str(speaker_id)+": " + transcribe(segment) +"\n"
    return text




import chinh_ta
def transcribe(wav):
    input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.batch_decode(pred_ids)[0]
    return pred_transcript

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model.to(device)
link = "/home/hungha/AI_365/phantachgiong/09_39_25-25_11_2023-204-TO-0357836690.wav"
# wav, _ = librosa.load(link, sr=16000)
# text = transcribe(wav)
# print(text)
# Example usage:
rttm_file_path = "/home/hungha/AI_365/phantachgiong/tmp_/pred_rttms/09_39_25-25_11_2023-204-TO-0357836690.rttm"
audio_file_path = "/home/hungha/AI_365/phantachgiong/tmp_/09_39_25-25_11_2023-204-TO-0357836690.wav"


speaker_intervals = extract_speaker_time_intervals(rttm_file_path)
t_=cut_and_save_audio(audio_file_path, speaker_intervals)
print(t_)