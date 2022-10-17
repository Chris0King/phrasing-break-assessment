import json
import os
import time
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from IPython.display import Audio
# from tqdm.notebook import tqdm
from time import sleep
from sklearn.model_selection import train_test_split
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--json_folder', type=str, default='')
parser.add_argument('--upload_folder', type=str, default='')
parser.add_argument('--res_path', type=str, default='')
opt = parser.parse_args()
os.makedirs(opt.upload_folder, exist_ok=True)

# folder path
wav_folder = 'dataset/l2_data/Wave'
wav_script_path = 'dataset/l2_data/text.txt'
result_folder = 'dataset/l2_data/json'
lj_res_folder = 'dataset/l1_data/json'
res_root_path = 'train'
# constant
DUR_TO_SEC = 1 / 10000000
DUR_TO_FRAME = 1 / 100000


def trans_to_bin(frame):
    if frame == 1:
        return '0'
    elif frame >= 3 and frame <= 5:
        return '1'
    elif frame <= 20:
        return '2'
    else:
        return '3'


def token_generate(json_path, wavid):
    item = {}
    item['breaks'] = []
    item['speeds'] = 0
    with open(json_path, encoding='utf8') as f:
        durations_info = json.load(f)
        words = durations_info['wordInfos']
        # get syllable size
        syllable_size = len(words[0]['syllInfo']) if len(words) > 0 else 0
        syl_duration = words[0]['Duration'] if len(words) > 0 else 0
        sent_token = words[0]['Word'] if len(words) > 0 else 0
        raw_data = sent_token
        for indx in range(1, len(words)):
            break_frame = (words[indx]['startTime'] -
                           words[indx - 1]['endTime']) * DUR_TO_FRAME
            # get all breaks
            item['breaks'].append(break_frame)
            syllable_size = syllable_size + len(words[indx]['syllInfo'])
            syl_duration = syl_duration + words[indx]['Duration']
            raw_data = raw_data + '<' + str(
                int(break_frame)) + '>' + words[indx]['Word']
            sent_token = sent_token + '<' + trans_to_bin(
                int(break_frame)) + '>' + words[indx]['Word']
    item['speeds'] = syllable_size / (syl_duration * DUR_TO_SEC)
    return [raw_data, sent_token, wavid]


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes = 100)
    dataset = {'wavid': [], 'raw_token': [], 'token': [], 'label': []}
    all_json = os.listdir(opt.json_folder)
    count = 0
    tokens = []
    for json_name in all_json:
        json_path = os.path.join(opt.json_folder, json_name)
        wavid = json_name.split('.')[0]
        # raw_data, sent_token = token_generate(json_path)
        # print sentence token: word1+<br>+word2
        res = pool.apply_async(token_generate, args=(json_path, wavid,)) #apply_sync的结果就是异步获取func的返回值
        tokens.append(res) #从异步提交任务获取结果
    for i in tokens:
        i = i.get()
        dataset['wavid'].append(i[2])
        dataset['raw_token'].append(i[0])
        dataset['token'].append(i[1])
        dataset['label'].append(1)
    df = pd.DataFrame(dataset)
    df.to_csv(opt.res_path, index=False)