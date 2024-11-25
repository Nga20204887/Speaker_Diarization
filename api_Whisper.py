import os
import wget
from omegaconf import OmegaConf
import json
import shutil
from faster_whisper import WhisperModel
import whisperx
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
# from deepmultilingualpunctuation import PunctuationModel
import torch
import re
import logging
import nltk
from utils import*
from flask import Flask, jsonify, send_file,request
import requests

def phantachhoithoai(vocal_target):

  file_name = os.path.basename(vocal_target)
  new_file_path = os.path.join("/home/hungha/AI_365/phantachgiong/results", file_name)

  # Whether to enable music removal from speech, helps increase diarization quality but uses alot of ram
  enable_stemming = True

  suppress_numerals = True

  if suppress_numerals:
      numeral_symbol_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
  else:
      numeral_symbol_tokens = None

  segments, info = whisper_model.transcribe(
      vocal_target,
      beam_size=5,
      word_timestamps=True,
      suppress_tokens=numeral_symbol_tokens,
      vad_filter=True,
  )
  whisper_results = []
  for segment in segments:
      whisper_results.append(segment._asdict())
  # clear gpu vram
  # del whisper_model

  print("ok segments")
  if info.language in wav2vec2_langs:
      device = "cpu"
      alignment_model, metadata = whisperx.load_align_model(
          language_code=info.language, device=device
      )
      result_aligned = whisperx.align(
          whisper_results, alignment_model, metadata, vocal_target, device
      )
      word_timestamps = filter_missing_timestamps(result_aligned["word_segments"])

      # clear gpu vram
      del alignment_model
      # torch.cuda.empty_cache()
  else:
      word_timestamps = []
      for segment in whisper_results:
          for word in segment["words"]:
              word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})
  print("ok word_timestamps")            
  temp_path ="/home/hungha/AI_365/phantachgiong/tmp"
  # os.makedirs(temp_path, exist_ok=True)
  config =create_config(temp_path,file_name)
  # shutil.copy(vocal_target, temp_path)
  # Initialize NeMo MSDD diarization model


  # msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(torch.device("cuda"))
  msdd_model = NeuralDiarizer(cfg=config)
  msdd_model.diarize()

  del msdd_model
  # Reading timestamps <> Speaker Labels mapping

  speaker_ts = []
  with open(os.path.join(temp_path, "pred_rttms", f"{os.path.splitext(file_name)[0]}.rttm"), "r") as f:
      lines = f.readlines()
      for line in lines:
          line_list = line.split(" ")
          s = int(float(line_list[5]) * 1000)
          e = s + int(float(line_list[8]) * 1000)
          speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

  wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

  ssm = get_sentences_speaker_mapping(wsm, speaker_ts)


  with open(f"{new_file_path}.txt", "w", encoding="utf-8-sig") as f:
      get_speaker_aware_transcript(ssm, f)

  with open(f"{new_file_path}.srt", "w", encoding="utf-8-sig") as srt:
      write_srt(ssm, srt)

  

import time

app = Flask(__name__)
@app.route('/diarization', methods=['POST', 'GET'])
def diarization():
  audio_url = request.values['link_audio']
  file_name = os.path.basename(audio_url)
  response = requests.get(audio_url)
  temp_path ="/home/hungha/AI_365/phantachgiong/tmp"
  os.makedirs(temp_path, exist_ok=True)
  file_path = os.path.join("/home/hungha/AI_365/phantachgiong/tmp", file_name)
  new_file_path = os.path.join("/home/hungha/AI_365/phantachgiong/results", file_name)
  print(file_path)
  open(file_path, "wb+").write(response.content)
  phantachhoithoai(file_path)
  print("success")
  cleanup(temp_path)
  with open(f"{new_file_path}.txt", "r", encoding="utf-8-sig") as file:
      content = file.read()
  return jsonify({'content': content})

if __name__ == '__main__':
    whisper_model_name = "large-v2"
    whisper_model = WhisperModel(whisper_model_name, device="cpu", compute_type="int8")
    app.run(debug= False, host='0.0.0.0', port=9001)
