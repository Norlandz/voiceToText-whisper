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

platform_system = platform.system()
print(platform_system)
print(platform.release())


############

import re
from dataclasses import asdict

import whisper
import whisper.model

# import ffmpeg

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


class VoiceToTextUtil:
    @staticmethod
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

    @staticmethod
    def get_fix_WordSegmentTimeMisAlign(segment_info: SegmentInfo):
        time_start_segment = segment_info["start"]
        if segment_info["words"].__len__() == 0:
            return time_start_segment

        time_start_word_1st = segment_info["words"][0]["start"]
        # sometimes can happen, forget what align or result thing
        if time_start_word_1st < time_start_segment:
            time_start_segment = time_start_word_1st
        return time_start_segment

    @staticmethod
    def write_file_WithOverwriteWarn(path: Path, content: str):
        print(f'{dt.now().strftime("%H:%M:%S")} writing text to: {path.__fspath__()}')
        try:
            with open(path, "x", encoding="utf-8") as file:
                file.write(content)
        except FileExistsError:
            content_old = None
            with open(path, "r+", encoding="utf-8") as file:
                content_old = file.read()
                file.seek(0)
                file.write(content)
                file.truncate()
            path_bak = path.parent / (path.name + ".bak")
            with open(path_bak, "w", encoding="utf-8") as file_bak:
                file_bak.write(content_old)


class VoiceToText_Whisper:

    def load_model_preconfig(self, modelSizeAndName: str, language: str, download_root: Path | None):
        #   |>"
        #   No, you do not need to load the Whisper model for each file. You can load the model once and then use it to process multiple files. Loading the model is typically a time-consuming operation, so reloading it for each file would be inefficient.
        #   <|
        #   https://chatgpt.com/c/4cf5d5ba-8ada-4f41-a279-25e8c6dbe394
        # |>"
        # Now consider the case where I want to call it like `alpha("FOO", myp2)` and `myp2` will either contain a value to be passed, or be `None`. But even though the function handles `p2=None`, I want it to use its default value `"bar"` instead.
        # <|
        # https://stackoverflow.com/questions/52494128/call-function-without-optional-arguments-if-they-are-none
        if download_root == None:
            model = whisper.load_model(name=modelSizeAndName)  # https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
        else:
            # |>"
            # > path.normalize('../../src/../src/node')
            # '../../src/node'
            # > path.resolve('../../src/../src/node')
            # '/Users/mtilley/src/node'
            # <|
            # https://stackoverflow.com/questions/10822574/difference-between-path-normalize-and-path-resolve-in-node-js
            model = whisper.load_model(name=modelSizeAndName, download_root=download_root.resolve().__fspath__())
        initial_prompt = """
    - put each complete sentence into each segment integrally, do not break and put an integral sentence into multiple segments, unless the sentence is too long.
    """
        print(f'{dt.now().strftime("%H:%M:%S")} ' + "model loaded")

        oDecodingOptions = whisper.DecodingOptions(
            # task='transcribe',
            language=language,
            # prompt='',
            # UserWarning: FP16 is not supported on CPU; using FP32 instead   warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            # fp16=False,  # ... idk, mem note gone
        )

        return model, initial_prompt, oDecodingOptions

    def voiceToText(self, pathFile_Audio: Path, model: whisper.Whisper, initial_prompt: str, oDecodingOptions: whisper.DecodingOptions):
        # pathStr_SubtitleVtt = re.sub(r"\.mp3$", ".vtt", pathFile_Audio.__fspath__())
        # pathStr_SubtitleVtt = re.sub(r"(.*)\.(.*?)$", r"\g<1>.vtt", pathFile_Audio.__fspath__())
        pathFolder_Output = pathFile_Audio.parent
        path_Vtt = pathFolder_Output / (pathFile_Audio.stem + ".vtt")
        path_Txt = pathFolder_Output / (pathFile_Audio.stem + ".txt")

        print(f'{dt.now().strftime("%H:%M:%S")} ' + pathFile_Audio.__fspath__())

        _transcribe_result = model.transcribe(pathFile_Audio.__fspath__(), word_timestamps=True, initial_prompt=initial_prompt, **asdict(oDecodingOptions))
        transcribe_result: TranscribeResult = _transcribe_result  # type: ignore
        # transcribe_result["text"]
        # transcribe_result["language"]
        print(f'{dt.now().strftime("%H:%M:%S")} ' + "transcribe done")

        if oDecodingOptions.language == None:
            raise Exception("oDecodingOptions.language == None")
        content_Vtt = (
            ##
            "WEBVTT"
            + "\nKind: captions"
            + "\nLanguage: "
            + oDecodingOptions.language
            # X-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000
            # Region: top
            # Style
            # ::cue {
            #     color: white;
            #     background-color: black;
            #     font-size: 1.2em;
            #     text-align: center;
            # }
        )

        content_Txt = ""

        for segment_info in transcribe_result["segments"]:
            timeVtt_start_segment = VoiceToTextUtil.format_ToVttTimestamp(VoiceToTextUtil.get_fix_WordSegmentTimeMisAlign(segment_info))
            timeVtt_end_segment = VoiceToTextUtil.format_ToVttTimestamp(segment_info["end"])
            segment_text = segment_info["text"]
            segment_id = segment_info["id"]
            print(f'{dt.now().strftime("%H:%M:%S")} ' + f"segment:: {segment_id:3} {segment_text}")
            # \n{segment_text[1:] if segment_text[0] == ' ' else segment_text}
            content_Vtt += f"\n\n{segment_id}\n{timeVtt_start_segment} --> {timeVtt_end_segment}\n"
            content_Txt += segment_text + "\n\n"
            for word_info in segment_info["words"]:
                timeVtt_start_word = VoiceToTextUtil.format_ToVttTimestamp(word_info["start"])
                timeVtt_end_word = VoiceToTextUtil.format_ToVttTimestamp(word_info["end"])
                word_text = word_info["word"]
                content_Vtt += f"<{timeVtt_start_word}><c>{word_text}</c>"
                # sys.exit(0) # DebugBreak
        print(f'{dt.now().strftime("%H:%M:%S")} ' + "content_vtt done")

        pathFolder_Output.mkdir(parents=True, exist_ok=True)
        VoiceToTextUtil.write_file_WithOverwriteWarn(path_Txt, content_Txt)
        VoiceToTextUtil.write_file_WithOverwriteWarn(path_Vtt, content_Vtt)

    def run(self, modelSizeAndName: str, language: str, pathFolder_AudioToTranscribe: Path, download_root: Path | None):
        model, initial_prompt, oDecodingOptions = self.load_model_preconfig(modelSizeAndName, language, download_root)

        for pathFile_Audio in pathFolder_AudioToTranscribe.iterdir():
            if pathFile_Audio.is_file():
                self.voiceToText(pathFile_Audio, model, initial_prompt, oDecodingOptions)


########
########
########

download_root = None

if platform_system == "Windows":  # Local Windows
    path_projectDir = path_entryScript_parentDir
    pathFolder_AudioToTranscribe = path_projectDir / "assets/AudioToTranscribe"
    download_root = path_projectDir / "model"
elif platform_system == "Linux":  # Google Colab
    # path_projectDir = cwd / "voiceToText-whisper"
    pathFolder_AudioToTranscribe = cwd
else:
    raise Exception("platform not supported")


########

# @config
modelSizeAndName = "base.en"  # tiny base small medium large
language = "en"  # zh None

########
oVoiceToText_Whisper = VoiceToText_Whisper()
oVoiceToText_Whisper.run(modelSizeAndName, language, pathFolder_AudioToTranscribe, download_root)
