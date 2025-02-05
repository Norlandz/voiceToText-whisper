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

import torch

has_cuda = torch.cuda.is_available()
device = torch.device("cuda" if has_cuda else "cpu")
print(device)

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

from typing import Optional, TypedDict, List


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


class RedoException(Exception):
    pass


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
            model = whisper.load_model(name=modelSizeAndName, device=device)  # https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
        else:
            # |>"
            # > path.normalize('../../src/../src/node')
            # '../../src/node'
            # > path.resolve('../../src/../src/node')
            # '/Users/mtilley/src/node'
            # <|
            # https://stackoverflow.com/questions/10822574/difference-between-path-normalize-and-path-resolve-in-node-js
            model = whisper.load_model(name=modelSizeAndName, device=device, download_root=download_root.resolve().__fspath__())
        initial_prompt = WhisperConfig.initial_prompt
        print(f'{dt.now().strftime("%H:%M:%S")} ' + "model loaded")

        oDecodingOptions = whisper.DecodingOptions(
            # task='transcribe',
            language=language,
            # prompt='',
            # UserWarning: FP16 is not supported on CPU; using FP32 instead   warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            # fp16=False,  # ... idk, mem note gone
            fp16=has_cuda, 
        )

        return model, initial_prompt, oDecodingOptions

    def voiceToText(self, pathFile_Audio: Path, model: whisper.Whisper, initial_prompt: str, oDecodingOptions: whisper.DecodingOptions, compression_ratio_threshold: float, modelSizeAndName: str):
        # pathStr_SubtitleVtt = re.sub(r"\.mp3$", ".vtt", pathFile_Audio.__fspath__())
        # pathStr_SubtitleVtt = re.sub(r"(.*)\.(.*?)$", r"\g<1>.vtt", pathFile_Audio.__fspath__())
        pathFolder_Output = pathFile_Audio.parent
        path_Vtt = pathFolder_Output / (pathFile_Audio.stem + ".vtt")
        path_Txt = pathFolder_Output / (pathFile_Audio.stem + ".txt")
        path_Fail = pathFolder_Output / (pathFile_Audio.stem + ".fail")

        print(f'{dt.now().strftime("%H:%M:%S")} ' + pathFile_Audio.__fspath__())

        if oDecodingOptions.language == None:
            raise Exception("oDecodingOptions.language == None")

        _transcribe_result = model.transcribe(
            pathFile_Audio.__fspath__(), word_timestamps=True, initial_prompt=initial_prompt, **asdict(oDecodingOptions), verbose=False, compression_ratio_threshold=compression_ratio_threshold
        )
        transcribe_result: TranscribeResult = _transcribe_result  # type: ignore
        # transcribe_result["text"]
        # transcribe_result["language"]
        print(f'{dt.now().strftime("%H:%M:%S")} ' + "transcribe done")

        content_Vtt = [
            (
                ##
                "WEBVTT"
                + "\nKind: captions"
                + f"\nLanguage: {oDecodingOptions.language}"
                + "\nNOTE"
                + f"\nmodelName: {modelSizeAndName}"
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
        ]

        content_Txt = ""

        requestRedoCount = 0
        segment_text_Prev = None
        for segment_info in transcribe_result["segments"]:
            timeVtt_start_segment = VoiceToTextUtil.format_ToVttTimestamp(VoiceToTextUtil.get_fix_WordSegmentTimeMisAlign(segment_info))
            timeVtt_end_segment = VoiceToTextUtil.format_ToVttTimestamp(segment_info["end"])
            segment_text = segment_info["text"]
            if segment_text_Prev is not None and VoiceToText_Whisper.check_RepeatTranscript(segment_text, segment_text_Prev):
                print(f'{dt.now().strftime("%H:%M:%S")} ' + f"segment:: {segment_id:3} {segment_text}")
                requestRedoCount += 1
            segment_id = segment_info["id"]
            if WhisperConfig.printTranscriptSegment:
                print(f'{dt.now().strftime("%H:%M:%S")} ' + f"segment:: {segment_id:3} {segment_text}")
            # \n{segment_text[1:] if segment_text[0] == ' ' else segment_text}
            content_Vtt.append(f"\n\n{segment_id}\n{timeVtt_start_segment} --> ")
            content_Vtt.append(f"{timeVtt_end_segment}\n")
            content_Txt += segment_text + "\n\n"
            wordCount_ForLineBreak = 0
            segment_FormattedWords = ""
            word_text_Prev = None
            for word_info in segment_info["words"]:
                wordCount_ForLineBreak += 1
                timeVtt_start_word = VoiceToTextUtil.format_ToVttTimestamp(word_info["start"])
                # timeVtt_end_word = VoiceToTextUtil.format_ToVttTimestamp(word_info["end"])
                word_text = word_info["word"]
                if wordCount_ForLineBreak > WhisperConfig.maxApproxWordsPerSegment:
                    if word_text_Prev is None:
                        raise ValueError("Impossible")
                    if re.search(r"\.|,|\?|!|;|。|，|？|！|；", word_text_Prev, re.IGNORECASE):
                        # content_Vtt = content_Vtt[:-13] # length of timeVtt_end_segment + \n
                        content_Vtt[-1] = f"{timeVtt_start_word}\n"
                        content_Vtt.append(segment_FormattedWords)
                        content_Vtt.append(f"\n\n{segment_id}\n{timeVtt_start_word} --> ")
                        content_Vtt.append(f"{timeVtt_end_segment}\n")
                        wordCount_ForLineBreak = 0
                        segment_FormattedWords = ""
                segment_FormattedWords += f"<{timeVtt_start_word}><c>{word_text}</c>"
                word_text_Prev = word_text
                # sys.exit(0) # DebugBreak
            content_Vtt.append(segment_FormattedWords)
            segment_text_Prev = segment_text
        print(f'{dt.now().strftime("%H:%M:%S")} ' + "content_vtt done")
        pathFolder_Output.mkdir(parents=True, exist_ok=True)
        VoiceToTextUtil.write_file_WithOverwriteWarn(path_Txt, content_Txt)
        VoiceToTextUtil.write_file_WithOverwriteWarn(path_Vtt, "".join(content_Vtt))
        if requestRedoCount >= 7:
            VoiceToTextUtil.write_file_WithOverwriteWarn(path_Fail, "")
            raise RedoException(f"segment_text - redo {requestRedoCount}: {segment_text}")

    @staticmethod
    def check_RepeatTranscript(text: str, text_Prev: str):
        """
        Checks if the text has 20 or more repeating special symbols, excluding Chinese characters.
        Check if the sentence is repeated.

        Args:
            text: The text string to check.

        Returns:
            True if the condition is met, False otherwise.
        """
        if text.strip() == text_Prev.strip():
            return True

        # Define a regular expression to find repeating special characters, excluding Chinese
        pattern = r"([^\w\s\u4e00-\u9fff])\1{19,}"  # (.+?)\1{12,}  dk speed ...
        match = re.search(pattern, text)
        return bool(match)

    def run(
        self,
        modelSizeAndName: str,
        language: str,
        pathFolder_AudioToTranscribe: Path,
        download_root: Path | None,
        filename_Regex: re.Pattern,
        list_filename_Include: list[str] | None,
        list_filename_Exclude: List[str],
    ):
        model_main, initial_prompt, oDecodingOptions = self.load_model_preconfig("medium", language, download_root)
        # if language == "zh":
        # model_alter, _, _ = self.load_model_preconfig("small", language, download_root)

        for pathFile_Audio in pathFolder_AudioToTranscribe.iterdir():
            if pathFile_Audio.is_file():
                if pathFile_Audio.name in list_filename_Exclude:
                    continue

                if filename_Regex.search(pathFile_Audio.name):
                    vtt_file_path = pathFile_Audio.with_suffix(".vtt")
                    if vtt_file_path.exists():
                        print(f'{dt.now().strftime("%H:%M:%S")} ' + " skipping existing vtt: " + pathFile_Audio.__fspath__())
                        continue

                    if not ((list_filename_Include is None) or (pathFile_Audio.name in list_filename_Include)):
                        continue

                    tryCount = 0
                    compression_ratio_threshold = 2.4
                    model = model_main
                    while tryCount <= 1:  # 2
                        tryCount += 1
                        try:
                            self.voiceToText(pathFile_Audio, model, initial_prompt, oDecodingOptions, compression_ratio_threshold, "notime")
                            break
                        except RedoException:
                            # compression_ratio_threshold -= 2.4
                            # model = model_alter
                            print(f'{dt.now().strftime("%H:%M:%S")} ' + f" redo-{tryCount} file: " + pathFile_Audio.__fspath__())
                    else:
                        print(f'{dt.now().strftime("%H:%M:%S")} ' + " failed file: " + pathFile_Audio.__fspath__())


########
########
########

# What is the difference between prompt and initial_prompt · openai/whisper · Discussion #1189
# https://github.com/openai/whisper/discussions/1189
#
# Is prompt same as prefix? · openai/whisper · Discussion #1080
# https://github.com/openai/whisper/discussions/1080
#
# prompt vs prefix in DecodingOptions · openai/whisper · Discussion #117
# https://github.com/openai/whisper/discussions/117#discussioncomment-3727051

# python - Cuda and OpenAI Whisper : enforcing GPU instead of CPU not working? - Stack Overflow
# https://stackoverflow.com/questions/75775272/cuda-and-openai-whisper-enforcing-gpu-instead-of-cpu-not-working
#
# How to get the progress bar while transcribing? · openai/whisper · Discussion #850
# https://github.com/openai/whisper/discussions/850
#
# --word_timestamps True lead to Failed to launch Triton kernels warning · openai/whisper · Discussion #1283
# https://github.com/openai/whisper/discussions/1283

# Some wired repetition happens in transcription. · openai/whisper · Discussion #192
# https://github.com/openai/whisper/discussions/192

download_root = None
pathFolder_AudioToTranscribe = None
filename_Regex = re.compile(r"\.(mp4|mov|webm|mkv|flv|avi|aac|mp3|m4a|wav)$")
list_filename_Include = None
list_filename_Exclude = []

if platform_system == "Windows":  # Local Windows
    path_projectDir = path_entryScript_parentDir
    pathFolder_AudioToTranscribe = path_projectDir / "assets/AudioToTranscribe"
    download_root = path_projectDir / "model"
elif platform_system == "Linux":  # Google Colab
    # path_projectDir = cwd / "voiceToText-whisper"
    pathFolder_AudioToTranscribe = cwd
else:
    raise Exception("platform not supported")


class WhisperConfig:
    printTranscriptSegment = False
    maxApproxWordsPerSegment = 15
    # - In some cases, you can consider a comma as a sentence break instead of a period.
    # - In some cases, you can break the sentence when the sentence is too long.
    # - For Chinese, each sentence is consider long when its over 20 or 40 words.
    # - Do not break in the middle of an integral sentence into smaller segments. Unless its longer than 20 words.
    # - Put each complete sentence into each segment integrally, ie do not break an integral sentence into multiple segments.
    # - Break the sentence appropriately at the comma or period, when its word count is over about 20 or 30.
    # - The audio should recognized the word "Bean" instead of "兵" or "病"
    ################
    # - No need to translate the video, just leave them as it.
    # - Do not repeat the same words too many times. Listen carefully.
    # - This video is about developing strategies with sql, database, Mysql.
    # - Some Chinese terminologies: 索引, B+树, B树, 叶节点, 节点, ...
    # - Some English terminologies in programming. eg: sql, mysql, MyISAM, InnoDB, database, tree, B-tree, B+ tree, skip list, value, index ...
# - This video is about developing a program with Java SpringBoot. 
# - The language of this video is Chinese, but there are few English terminologies in programming. eg: Spring, SpringBoot, Boot, Bean, Context, Application, SpringMVC, Controller, Model, User, Post, Get, Database, REST, JSON, Request, Body, Maven, POM, View, Dependency, Configuration, Test, Environment, RabbitMQ, MongoDB, Tomcat, Starter, ...
# - This video is about developing a program with JavaScript Vue3. 
    initial_prompt = """
- 本视频关于 JavaScript Vue Uniapp.
"""


########

# @config
# modelSizeAndName = "base.en"  # tiny base small medium large
# language = "en"  # zh None
modelSizeAndName = "medium"  # small seems ok; medium little better on English?... guess medium.. ;
language = "zh"  # zh None
# pathFolder_AudioToTranscribe = Path(r"C:/usp/usehpsj/study/Spring_2021_10_HeiMa/黑马程序员SpringBoot2全套视频教程，springboot零基础到项目实战（spring boot2完整版）")
# pathFolder_AudioToTranscribe = Path(r"C:/usp/usehpsj/study/SpringCloud_2024_04_HeiMa/黑马程序员SpringCloud微服务开发与实战，java黑马商城项目微服务实战开发（涵盖MybatisPlus、Docker、MQ、ES、Redis高级等）")
# pathFolder_AudioToTranscribe = Path(r"C:/usp/usehpsj/study/sql_Interview/2019年阿里数据库索引面试题，100分钟讲透MySQL索引底层原理！")
# pathFolder_AudioToTranscribe = Path(r"C:/usp/usehpsj/study/sql_Interview/尚硅谷MySQL数据库面试题宝典，mysql面试必考！mysql工作必用！")
# pathFolder_AudioToTranscribe = Path(r"C:\usp\usehpsj\study\sql_Interview\黑马程序员 MySQL数据库入门到精通，从mysql安装到mysql高级、mysql优化全囊括")
# pathFolder_AudioToTranscribe = Path(r"C:\usp\usehpsj\study\_new_bilibili\黑马程序员Spring视频教程，深度讲解spring5底层原理")
# pathFolder_AudioToTranscribe = Path(r"C:\usp\usehpsj\study\SpringCloud_2024_04_HeiMa\黑马程序员SpringBoot3+Vue3全套视频教程，springboot+vue企业级全栈开发从基础、实战到面试一套通关")
# pathFolder_AudioToTranscribe = Path(r"C:\usp\usehpsj\study\_new_bilibili\尚硅谷Vue3入门到实战，最新版vue3+TypeScript前端开发教程")
# pathFolder_AudioToTranscribe = Path(r"C:\usp\usehpsj\study\_new_bilibili\黑马程序员前端项目uniapp小兔鲜儿微信小程序项目视频教程，基于Vue3+Ts+Pinia+uni-app的最新组合技术栈开发的电商业务全流程")
# pathFolder_AudioToTranscribe = Path(r"C:\usp\usehpsj\study\_new_bilibili\【已更新】24年前端面试题八股文（Vue、js、css、h5c3、echarts、uniapp、webpack、git、hr）")
# pathFolder_AudioToTranscribe = Path(r"C:\usp\usehpsj\study\_new_bilibili\黑马程序员前端微信小程序开发教程，微信小程序从基础到发布全流程_企业级商城实战(含uni-app项目多端部署)")
pathFolder_AudioToTranscribe = Path(r"C:\usp\usehpsj\study\_new_bilibili\黑马程序员人工智能教程_10小时学会图像处理OpenCV入门教程")
# filename_Regex = re.compile(r"^\[P(\d+)(.*?)\.mp4$")
# list_filename_Include = [
    # r"[P095]实战篇-83_大事件_顶部导航栏_下拉菜单功能实现",
    # r"[P096]实战篇-84_大事件_基本资料修改",
    # r"[P097]实战篇-85_大事件_用户头像修改",
# ]
# list_filename_Exclude = [
# ]

########
oVoiceToText_Whisper = VoiceToText_Whisper()
oVoiceToText_Whisper.run(modelSizeAndName, language, pathFolder_AudioToTranscribe, download_root, filename_Regex, list_filename_Include, list_filename_Exclude)
