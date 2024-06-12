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
from utils import*
import chinh_ta
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import requests
from threading import Semaphore
lock = Semaphore(4)

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
def merge_intervals(intervals):
    merged_intervals = []
    current_interval = intervals[0]

    for interval in intervals[1:]:
        if interval[2] == current_interval[2]:
            # Merge intervals with the same speaker ID
            current_interval = (current_interval[0], interval[1], current_interval[2])
        else:
            # Add the current merged interval to the list
            merged_intervals.append(current_interval)
            # Move to the next interval
            current_interval = interval

    # Add the last merged interval
    merged_intervals.append(current_interval)

    return merged_intervals
def cut_and_transcribe_audio(input_audio_path,  intervals, sr=16000):
    audio, sr = librosa.load(input_audio_path, sr=sr)
    text =""
    
    for i, (start, end, speaker_id) in enumerate(intervals, 1):
        # Convert start and end times to sample indices
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        t = ""
        # Extract the segment
        segment = audio[start_idx:end_idx]
        if((end- start)>240):
            segment_length_sec = (end- start)
            # Trim audio to remove silence at the beginning and end
            trimmed_audio, _ = librosa.effects.trim(segment)

            # Calculate segment length in samples
            segment_length_samples = int(segment_length_sec * sr)

            # Calculate the number of segments
            num_segments = len(trimmed_audio) // segment_length_samples

            # Cut audio into segments
            audio_segments = [trimmed_audio[i * segment_length_samples:(i + 1) * segment_length_samples] for i in range(num_segments)]
            for audio in audio_segments:
                t += transcribe(audio) + " "
        else:
            t = transcribe(segment)
        if(t != ""):  
            text +=  str(speaker_id)+": " + t +"\n"
    return text

def transcribe(wav):
    input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.batch_decode(pred_ids)[0]
    return pred_transcript
  
def phantachhoithoai(vocal_target,temp_path,name):
    if (name == 1):
        # vocal_target = "/home/hungha/AI_365/phantachgiong/09_39_25-25_11_2023-204-TO-0357836690.wav"
        file_name1 = os.path.basename(vocal_target)
        
        # os.makedirs(temp_path, exist_ok=True)
        config1 =create_config(temp_path,file_name1)


        # msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(torch.device("cuda"))
        msdd_model1 = NeuralDiarizer(cfg=config1)
        msdd_model1.diarize()
        del msdd_model1
        rttm_file_path1 = os.path.join(f"{temp_path}/pred_rttms", f"{os.path.splitext(file_name1)[0]}.rttm")
        speaker_intervals1 = extract_speaker_time_intervals(rttm_file_path1)
        merged_intervals1 = merge_intervals(speaker_intervals1)
        t_1 = cut_and_transcribe_audio(vocal_target, merged_intervals1)
        print(t_1)
        print("----------------------------")
        with open('/home/hungha/AI_365/phantachgiong/dict2.txt', 'r', encoding='utf-8') as dic1:
            lines1 = dic1.readlines()
            for line1 in lines1:
                x1 = line1.split('|')
                # print(x)
                t_1 = re.sub(x1[0], x1[1][:-1], t_1)
        split_strings1 = t_1.split('\n')

        processed_strings1 = []
        
        
        for string1 in split_strings1:
            # Xử lý chuỗi con ở đây (ví dụ: loại bỏ khoảng trắng thừa)

            string1 = chinh_ta.terminal_input(string1)
            processed_strings1.append(string1)
    
        # Hợp các chuỗi đã xử lý thành một chuỗi mới, cách nhau bởi kí tự '\n'
        
        t_1 = '\n'.join(filter(None, processed_strings1))
        return t_1
    elif(name == 2):
        # vocal_target = "/home/hungha/AI_365/phantachgiong/09_39_25-25_11_2023-204-TO-0357836690.wav"
        file_name2 = os.path.basename(vocal_target)
        
        # os.makedirs(temp_path, exist_ok=True)
        config2 =create_config(temp_path,file_name2)


        # msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(torch.device("cuda"))
        msdd_model2 = NeuralDiarizer(cfg=config2)
        msdd_model2.diarize()
        del msdd_model2
        rttm_file_path2 = os.path.join(f"{temp_path}/pred_rttms", f"{os.path.splitext(file_name2)[0]}.rttm")
        speaker_intervals2 = extract_speaker_time_intervals(rttm_file_path2)
        merged_intervals2 = merge_intervals(speaker_intervals2)
        t_2 = cut_and_transcribe_audio(vocal_target, merged_intervals2)
        print(t_2)
        print("----------------------------")
        with open('/home/hungha/AI_365/phantachgiong/dict2.txt', 'r', encoding='utf-8') as dic2:
            lines2 = dic2.readlines()
            for line2 in lines2:
                x2 = line2.split('|')
                # print(x)
                t_2 = re.sub(x2[0], x2[1][:-1], t_2)
        split_strings2 = t_2.split('\n')

        processed_strings2 = []
        
        
        for string2 in split_strings2:
            # Xử lý chuỗi con ở đây (ví dụ: loại bỏ khoảng trắng thừa)

            string2 = chinh_ta.terminal_input(string2)
            processed_strings2.append(string2)
    
        # Hợp các chuỗi đã xử lý thành một chuỗi mới, cách nhau bởi kí tự '\n'
        
        t_2 = '\n'.join(filter(None, processed_strings2))
        return t_2
    elif(name == 3):
        # vocal_target = "/home/hungha/AI_365/phantachgiong/09_39_25-25_11_2023-204-TO-0357836690.wav"
        file_name3 = os.path.basename(vocal_target)
        
        # os.makedirs(temp_path, exist_ok=True)
        config3 =create_config(temp_path,file_name3)


        # msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(torch.device("cuda"))
        msdd_model3 = NeuralDiarizer(cfg=config3)
        msdd_model3.diarize()
        del msdd_model3
        rttm_file_path3 = os.path.join(f"{temp_path}/pred_rttms", f"{os.path.splitext(file_name3)[0]}.rttm")
        speaker_intervals3 = extract_speaker_time_intervals(rttm_file_path3)
        merged_intervals3 = merge_intervals(speaker_intervals3)
        t_3 = cut_and_transcribe_audio(vocal_target, merged_intervals3)
        print(t_3)
        print("----------------------------")
        with open('/home/hungha/AI_365/phantachgiong/dict2.txt', 'r', encoding='utf-8') as dic3:
            lines3 = dic3.readlines()
            for line3 in lines3:
                x3 = line3.split('|')
                # print(x)
                t_3 = re.sub(x3[0], x3[1][:-1], t_3)
        split_strings3 = t_3.split('\n')

        processed_strings3 = []
        
        
        for string3 in split_strings3:
            # Xử lý chuỗi con ở đây (ví dụ: loại bỏ khoảng trắng thừa)

            string3 = chinh_ta.terminal_input(string3)
            processed_strings3.append(string3)
    
        # Hợp các chuỗi đã xử lý thành một chuỗi mới, cách nhau bởi kí tự '\n'
        
        t_3 = '\n'.join(filter(None, processed_strings3))
        return t_3
        
    else:
        # vocal_target = "/home/hungha/AI_365/phantachgiong/09_39_25-25_11_2023-204-TO-0357836690.wav"
        file_name = os.path.basename(vocal_target)
        
        # os.makedirs(temp_path, exist_ok=True)
        config =create_config(temp_path,file_name)


        # msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(torch.device("cuda"))
        msdd_model = NeuralDiarizer(cfg=config)
        msdd_model.diarize()
        del msdd_model
        rttm_file_path = os.path.join(f"{temp_path}/pred_rttms", f"{os.path.splitext(file_name)[0]}.rttm")
        speaker_intervals = extract_speaker_time_intervals(rttm_file_path)
        merged_intervals = merge_intervals(speaker_intervals)
        t_ = cut_and_transcribe_audio(vocal_target, merged_intervals)
        print(t_)
        print("----------------------------")
        with open('/home/hungha/AI_365/phantachgiong/dict2.txt', 'r', encoding='utf-8') as dic:
            lines = dic.readlines()
            for line in lines:
                x = line.split('|')
                # print(x)
                t_ = re.sub(x[0], x[1][:-1], t_)
        split_strings = t_.split('\n')

        processed_strings = []
        
        
        for string in split_strings:
            # Xử lý chuỗi con ở đây (ví dụ: loại bỏ khoảng trắng thừa)

            string = chinh_ta.terminal_input(string)
            processed_strings.append(string)
    
        # Hợp các chuỗi đã xử lý thành một chuỗi mới, cách nhau bởi kí tự '\n'
        
        t_ = '\n'.join(filter(None, processed_strings))
        print(t_)
        return t_
from pydub.exceptions import CouldntDecodeError

def is_corrupted_audio(file_path):
    try:
        AudioSegment.from_file(file_path)
        return True  # File âm thanh không bị hỏng
    except CouldntDecodeError as e:
        print(f"Audio file is corrupted: {e}")
        return False  # File âm thanh bị hỏng
app = Flask(__name__)


@app.route('/diarization', methods=['POST', 'GET'])
def diarization():
    global id_api 
    id_api = 0 
    while True:
        acquired = lock.acquire(blocking=False)
        if acquired:
            id_api += 1
            
            if((id_api %4) == 1):
                try:
                    audio_url1 = request.values['link_audio']
                    file_name1 = os.path.basename(audio_url1)
                    response1 = requests.get(audio_url1)
                    name_1 = os.path.splitext(file_name1)[0]
                    temp_path1 ="/home/hungha/AI_365/phantachgiong/tmp_"+ name_1
                    os.makedirs(temp_path1, exist_ok=True)
                    file_path1 = os.path.join(temp_path1, file_name1)
                    open(file_path1, "wb+").write(response1.content)
                    content1 = ""
                    if(is_corrupted_audio(file_path1)):
                        try:
                            content1 = phantachhoithoai(file_path1,temp_path1,1)
                            cleanup(temp_path1)
                            print("success")
                            return jsonify({'content': content1})
                            
                        except Exception as err:
                            cleanup(temp_path1)
                            print("err:", err )
                    else:
                        return jsonify({'content': "Audio file is corrupted."})

                finally:
                    # Release khóa lock sau khi xử lý hoàn tất
                    lock.release()
            elif((id_api %4) == 2):
                try:
                    audio_url2 = request.values['link_audio']
                    file_name2 = os.path.basename(audio_url2)
                    response2 = requests.get(audio_url2)
                    name_2 = os.path.splitext(file_name2)[0]
                    temp_path2 ="/home/hungha/AI_365/phantachgiong/tmp_"+ name_2
                    os.makedirs(temp_path2, exist_ok=True)
                    file_path2 = os.path.join(temp_path2, file_name2)
                    open(file_path2, "wb+").write(response2.content)
                    content2 = ""
                    if(is_corrupted_audio(file_path2)):
                        try:
                            content2 = phantachhoithoai(file_path2,temp_path2,2)
                            cleanup(temp_path2)
                            print("success")
                            return jsonify({'content': content2})
                            
                        except Exception as err:
                            cleanup(temp_path2)
                            print("err:", err )
                    else:
                        return jsonify({'content': "Audio file is corrupted."})

                finally:
                    # Release khóa lock sau khi xử lý hoàn tất
                    lock.release()
            elif((id_api %4) == 3):
                try:
                    audio_url3 = request.values['link_audio']
                    file_name3 = os.path.basename(audio_url3)
                    response3 = requests.get(audio_url3)
                    name_3 = os.path.splitext(file_name3)[0]
                    temp_path3 ="/home/hungha/AI_365/phantachgiong/tmp_"+ name_3
                    os.makedirs(temp_path3, exist_ok=True)
                    file_path3 = os.path.join(temp_path3, file_name3)
                    open(file_path3, "wb+").write(response3.content)
                    content3 = ""
                    if(is_corrupted_audio(file_path3)):
                        try:
                            content3 = phantachhoithoai(file_path3,temp_path3,3)
                            cleanup(temp_path3)
                            print("success")
                            return jsonify({'content': content3})
                            
                        except Exception as err:
                            cleanup(temp_path3)
                            print("err:", err )
                    else:
                        return jsonify({'content': "Audio file is corrupted."})


                finally:
                    # Release khóa lock sau khi xử lý hoàn tất
                    lock.release()
            else:
                try:
                    audio_url = request.values['link_audio']
                    file_name = os.path.basename(audio_url)
                    response = requests.get(audio_url)
                    name_ = os.path.splitext(file_name)[0]
                    temp_path ="/home/hungha/AI_365/phantachgiong/tmp_"+ name_
                    os.makedirs(temp_path, exist_ok=True)
                    file_path = os.path.join(temp_path, file_name)
                    open(file_path, "wb+").write(response.content)
                    content = ""
                    if(is_corrupted_audio(file_path)):
                        try:
                            content = phantachhoithoai(file_path,temp_path,)
                            cleanup(temp_path)
                            print("success")
                            return jsonify({'content': content})
                            
                        except Exception as err:
                            cleanup(temp_path)
                            print("err:", err )
                    else:
                        return jsonify({'content': "Audio file is corrupted."})


                finally:
                    # Release khóa lock sau khi xử lý hoàn tất
                    lock.release()
            

        else:
            time.sleep(2)  # Đợi 1 giây trước khi thử lại
    


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    model.to(device)
    app.run(debug= False, host='43.239.223.184', port=9002)
