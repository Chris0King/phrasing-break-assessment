import json
import os
import time
import azure.cognitiveservices.speech as speechsdk
# $ pip install azure-cognitiveservices-speech
from azure_speech_args_config import azure_speech_args_config
import threading
from concurrent.futures import ThreadPoolExecutor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--wav_folder', type=str, default='../dataset/hi_fi_tts_v0/wavs')
parser.add_argument('--wav_script_path', type=str, default='../dataset/hi_fi_tts_v0/metadata.csv')
parser.add_argument('--json_folder', type=str, default='../dataset/hi_fi_tts_v0/json')
parser.add_argument('--upload_folder', type=str, default='../dataset/hi_fi_tts_v0')
opt = parser.parse_args()
os.makedirs(opt.upload_folder, exist_ok=True)

# constant
wav_folder = 'dataset/l2_data/Wave'
wav_script_path = 'dataset/l2_data/text.txt'
result_folder = 'dataset/l2_data/json'
DUR_TO_SEC = 1 / 10000000
DUR_TO_FRAME = 1 / 100000


def recognize(azure_speech_args_config, file_name, sentenceString):
    speech_config = speechsdk.SpeechConfig(
        subscription=azure_speech_args_config['speech_key'],
        region=azure_speech_args_config['service_region'])
    speech_config.request_word_level_timestamps()
    speech_config.output_format = speechsdk.OutputFormat(1)
    audio_config = speechsdk.audio.AudioConfig(filename=file_name)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                                   audio_config=audio_config)
    connect = speechsdk.Connection.from_recognizer(speech_recognizer)
    connect.set_message_property(
        "speech.context", "phraseDetection",
        json.dumps({
            "enrichment": {
                "pronunciationAssessment": {
                    "referenceText": sentenceString,
                    "gradingSystem": "HundredMark",
                    "granularity": "Phoneme",
                    "dimension": "Comprehensive",
                    "enableMiscue": "true",
                    "scenarioId": "198a72a3-1e5a-43af-8b0c-fd4a2a0f62d1",
                    "nbestPhonemeCount": azure_speech_args_config['nbest'],
                    "phonemeAlphabet":
                    azure_speech_args_config['phon_alphabet']
                }
            }
        }))
    connect.set_message_property(
        "speech.context", "phraseOutput",
        json.dumps({
            "detailed": {
                "options": ["WordTimings", "PronunciationAssessment"]
            }
        }))
    return speech_recognizer.recognize_once().json


def getAzureSpeechStudentResult(azure_speech_args_config, wav_file_name,
                                sentence):
    resultStr = recognize(azure_speech_args_config, wav_file_name, sentence)
    return resultStr


def getAzureSpeechStudentDurationAndPhonemes(sentence, azure_speech_result):
    studentInfo = {}
    try:
        if len(azure_speech_result['NBest']) != 1:
            # print(wavid, 'azure speech output error')
            return {}
            raise KeyError('azure speech output error')
        else:
            studentInfo['recognitionStatus'] = azure_speech_result[
                'RecognitionStatus']
            studentInfo['startOffset'] = azure_speech_result['Offset']
            studentInfo['sentenceDuration'] = azure_speech_result['Duration']
            studentInfo['sentenceText'] = azure_speech_result['DisplayText']
            studentWordsInfo = []
            wordsInfo = azure_speech_result['NBest'][0]['Words']
            if len(wordsInfo) == len(sentence.split(" ")):
                for index in range(len(wordsInfo)):
                    wordInfoJson = wordsInfo[index]
                    studentSyllInfo = []
                    for syllIndex in range(len(wordInfoJson['Syllables'])):
                        syllInfoJson = wordInfoJson['Syllables'][syllIndex]
                        studentSyllInfo.append({
                            'Syllable':
                            syllInfoJson['Syllable'],
                            'Offset':
                            syllInfoJson['Offset'],
                            'Duration':
                            syllInfoJson['Duration'],
                            'startTime':
                            syllInfoJson['Offset'],
                            'endTime':
                            syllInfoJson['Offset'] + syllInfoJson['Duration'],
                        })
                    studentPhonemes = []
                    for phonemeIndex in range(len(wordInfoJson['Phonemes'])):
                        phonemeJson = wordInfoJson['Phonemes'][phonemeIndex]
                        studentPhonemes.append({
                            'Syllable':
                            phonemeJson['Phoneme'],
                            'Offset':
                            phonemeJson['Offset'],
                            'Duration':
                            phonemeJson['Duration'],
                            'startTime':
                            phonemeJson['Offset'],
                            'endTime':
                            phonemeJson['Offset'] + phonemeJson['Duration'],
                        })
                    studentWordsInfo.append({
                        'Word':
                        wordInfoJson['Word'],
                        'Offset':
                        wordInfoJson['Offset'],
                        'Duration':
                        wordInfoJson['Duration'],
                        'startTime':
                        wordInfoJson['Offset'],
                        'endTime':
                        wordInfoJson['Offset'] + wordInfoJson['Duration'],
                        'syllInfo':
                        studentSyllInfo,
                        'phonemes':
                        studentPhonemes
                    })
                studentInfo['wordInfos'] = studentWordsInfo
                return studentInfo
            else:
                # print(wavid, 'azure speech output error')
                return {}
                raise KeyError('azure speech output error')
    except KeyError as e:
        #         print(wavid, 'azure speech output error')
        return {}
        raise KeyError("azure speech output error")


def save_dura_info(sentence,
                   wavid,
                   wav_folder=wav_folder,
                   result_folder=result_folder,
                   filetype='.wav'):
    print(wavid, "start")
    if not os.path.exists(result_folder):
      os.makedirs(result_folder)
    json_path = os.path.join(result_folder, wavid + '.json')
    if os.path.exists(json_path):
        print(wavid, 'exist')
        return
    wav_path = os.path.join(wav_folder, wavid + filetype)
    if not os.path.exists(wav_path):
        print(wavid, 'wav not exist')
        return
    azure_result = getAzureSpeechStudentResult(azure_speech_args_config,
                                               wav_path, sentence)
    if azure_result:
        azure_result = json.loads(azure_result)
    else:
        print(wavid, "empty end")
        return
    durations_info = getAzureSpeechStudentDurationAndPhonemes(
        sentence, azure_result)
    if not durations_info:
        print('no duration info')
        return
    with open(json_path, 'w', encoding="utf-8") as f:
        json.dump(durations_info, f, ensure_ascii=False, indent=4)
    print(wavid, "end")

ls_script_path = 'l1_data/metadata.txt'
ls_res_folder = 'l1_data/json'
ls_wav_folder = 'l1_data/wavs'

count = 0
if __name__ == '__main__':
    max_connections = 100  # 定义最大线程数
    # with open(opt.wav_script_path, 'r', encoding='utf-8') as f:
    #     for line in (f.readlines()):
    #         # if count > 1000:
    #         #     break
    #         # count = count + 1
    #         line = line.strip().split('|')
    #         wavid = line[0]
    #         sentence = line[1]
    #         task = t.submit(save_dura_info, sentence,
    #                                 wavid,
    #                                 opt.wav_folder,
    #                                 opt.json_folder)
    with ThreadPoolExecutor(max_workers = max_connections) as t:  # 创建一个最大容纳数量为5的线程池
        with open(opt.wav_script_path, 'r', encoding='utf-8') as f:
            for line in (f.readlines()):
                # if count > 1000:
                #     break
                # count = count + 1
                line = line.strip().split('|')
                wavid = line[0]
                sentence = line[1]
                task = t.submit(save_dura_info, sentence,
                                     wavid,
                                     opt.wav_folder,
                                     opt.json_folder)