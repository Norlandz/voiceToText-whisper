import os
import sys
from pathlib import Path

from datetime import datetime as dt
from datetime import timedelta

print(f'{dt.now().strftime("%H:%M:%S")} ' + ">---<")
cwd = Path.cwd()
print(f'{dt.now().strftime("%H:%M:%S")} ' + cwd.__fspath__())
argv0 = sys.argv[0]
print(f'{dt.now().strftime("%H:%M:%S")} ' + argv0)
path_entryScript = Path(argv0).resolve()
print(f'{dt.now().strftime("%H:%M:%S")} ' + path_entryScript.__fspath__())
path_entryScript_parentDir = path_entryScript.parent

import platform

print(os.name)
print(sys.platform)

print(platform.system())
print(platform.release())

platform_system = platform.system()

############

import re
from dataclasses import asdict

import whisper.model
import whisper

# import ffmpeg

############


def format_ToVttTimestamp(seconds: float):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, int(seconds), milliseconds)


# def format_seconds(seconds):
#     delta = timedelta(seconds=seconds)
#     #     formatted_time = (timedelta(0) + delta).strftime('%H:%M:%S')
#     formatted_time = str(delta).split(".")[0]  # Remove microseconds part
#     return formatted_time
# print(format_seconds(64.02))  # Output: 0:01:04

############


# transcribe_result
#  {'text': ' The quick brown fox jumps over the lazy dog.', 'segments': [{...}], 'language': 'en'}
# segment_info | transcribe_result['segments']
#   {'id': 0, 'seek': 0, 'start': 0.9000000000000001, 'end': 4.34, 'text': ' The quick brown fox jumps over the lazy dog.', 'tokens': [50363, 383, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13], 'temperature': 0.0, 'avg_logprob': -0.07259444892406464, 'compression_ratio': 0.8627450980392157, 'no_speech_prob': 0.028506537899374962, 'words': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}]}
# word_info | segment_info['words']
#   [{'word': ' The', 'start': 0.9000000000000001, 'end': 1.64, 'probability': 0.9178125858306885}, {'word': ' quick', 'start': 1.64, 'end': 1.86, 'probability': 0.756718635559082}, {'word': ' brown', 'start': 1.86, 'end': 2.18, 'probability': 0.8945452570915222}, {'word': ' fox', 'start': 2.18, 'end': 2.52, 'probability': 0.957324206829071}, {'word': ' jumps', 'start': 2.52, 'end': 3.02, 'probability': 0.9977018237113953}, {'word': ' over', 'start': 3.02, 'end': 3.42, 'probability': 0.998938262462616}, {'word': ' the', 'start': 3.42, 'end': 3.58, 'probability': 0.9985789060592651}, {'word': ' lazy', 'start': 3.58, 'end': 3.82, 'probability': 0.9762164950370789}, {'word': ' dog.', 'start': 3.82, 'end': 4.34, 'probability': 0.9945038557052612}]

from typing import TypedDict, List


class WordInfo(TypedDict):
    word: str
    start: float
    end: float
    probability: float


class SegmentInfo(TypedDict):
    id: int
    seek: float
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: List[WordInfo]


class TranscribeResult(TypedDict):
    text: str
    segments: List[SegmentInfo]  # segments: List[dict]
    language: str


############

if platform_system == "Windows": # Local Windows
    path_projectDir = path_entryScript_parentDir
elif platform_system == "Linux": # Google Colab
    path_projectDir = cwd / "voiceToText-whisper"
else:
    raise Exception("platform not supported")

path_Audio = path_projectDir / "assets/0014163210622 - job viva packaging.mp3"
# path_Audio = "./assets/test.mp3"
pathStr_Subtitle = re.sub(r"\.mp3$", ".vtt", path_Audio.__fspath__())
print(f'{dt.now().strftime("%H:%M:%S")} ' + path_Audio.__fspath__())

model = whisper.load_model("base.en") # small medium
initial_prompt = """
- put each complete sentence into each segment integrally, do not break and put an integral sentence into multiple segments, unless the sentence is too long.
"""
print(f'{dt.now().strftime("%H:%M:%S")} ' + "model loaded")

oDecodingOptions = whisper.DecodingOptions(
    # task='transcribe',
    language="en",
    # prompt='',
    # UserWarning: FP16 is not supported on CPU; using FP32 instead   warnings.warn("FP16 is not supported on CPU; using FP32 instead")
    # fp16=False,  # ... idk, mem note gone
)
_transcribe_result = model.transcribe(path_Audio.__fspath__(), word_timestamps=True, initial_prompt=initial_prompt, **asdict(oDecodingOptions))
transcribe_result: TranscribeResult = _transcribe_result  # type: ignore
# transcribe_result["text"]
# transcribe_result["language"]
print(f'{dt.now().strftime("%H:%M:%S")} ' + "transcribe done")

content_vtt = """WEBVTT
Kind: captions
Language: en
"""
# X-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000
# Region: top
# Style
# ::cue {
#     color: white;
#     background-color: black;
#     font-size: 1.2em;
#     text-align: center;
# }


def get_fix_WordSegmentTimeMisAlign(segment_info: SegmentInfo):
    time_start_segment = segment_info["start"]
    if segment_info["words"].__len__() == 0:
        return time_start_segment

    time_start_word_1st = segment_info["words"][0]["start"]
    # sometimes can happen, forget what align or result thing
    if time_start_word_1st < time_start_segment:
        time_start_segment = time_start_word_1st
    return time_start_segment


for segment_info in transcribe_result["segments"]:
    timeVtt_start_segment = format_ToVttTimestamp(get_fix_WordSegmentTimeMisAlign(segment_info))
    timeVtt_end_segment = format_ToVttTimestamp(segment_info["end"])
    segment_text = segment_info["text"]
    segment_id = segment_info["id"]
    print(f'{dt.now().strftime("%H:%M:%S")} ' + f"segment:: {segment_id:3} {segment_text}")
    # \n{segment_text[1:] if segment_text[0] == ' ' else segment_text}
    content_vtt += f"\n\n{segment_id}\n{timeVtt_start_segment} --> {timeVtt_end_segment}\n"
    for word_info in segment_info["words"]:
        timeVtt_start_word = format_ToVttTimestamp(word_info["start"])
        timeVtt_end_word = format_ToVttTimestamp(word_info["end"])
        word_text = word_info["word"]
        content_vtt += f"<{timeVtt_start_word}><c>{word_text}</c>"
        # sys.exit(0) # DebugBreak
print(f'{dt.now().strftime("%H:%M:%S")} ' + "content_vtt done")

file_vtt = open(pathStr_Subtitle, "w", encoding="utf-8")
file_vtt.write(content_vtt)
