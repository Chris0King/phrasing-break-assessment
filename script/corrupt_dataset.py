import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--corrupt_rate', type=int, default=3)
parser.add_argument('--corrupt_chance', type=float, default=0.15)
parser.add_argument('--token_path', type=str, default='')
parser.add_argument('--upload_folder', type=str, default='')
parser.add_argument('--res_root_path', type=str, default='')
parser.add_argument('--test_size', type=int, default=2000)
opt = parser.parse_args()
os.makedirs(opt.upload_folder, exist_ok=True)

l1_df = pd.read_csv(opt.token_path)
l1_df['label'] = 0
l2_size = int(opt.corrupt_rate * l1_df.shape[0])


# break_type: 0,1,2,3
def change_break_type(break_type):
    break_list = [0, 1, 2, 3]
    chance = opt.corrupt_chance / 3
    p = np.array([chance, chance, chance, chance])
    p[break_type] = 1 - 3 * chance
    return np.random.choice(break_list, p=p.ravel())


def corrupt_data(sentence):
    sentence = sentence.split('<')
    corrupted_sent = sentence[0]
    for item in sentence[1:]:
        changed_type = change_break_type(int(item[0]))
        corrupted_sent = corrupted_sent + '<' + str(changed_type) + item[1:]
    if sentence == corrupt_data:
        return corrupt_data(sentence)
    return corrupted_sent


# corrupt dataset
corrupted_data = {'raw_token': [], 'token': [], 'label': []}
raw_l1_df = pd.concat([l1_df, l1_df, l1_df])
for index, row in raw_l1_df.iterrows():
    token = row['token']
    corrupted_data['raw_token'].append(row['raw_token'])
    corrupted_data['token'].append(corrupt_data(row['token']))
    corrupted_data['label'].append(1)
corrupted_df = pd.DataFrame(corrupted_data)
corrupted_df.sample(frac=1)

# split dataset
data = pd.concat([l1_df, corrupted_df[:l2_size]])
# data: 需要进行分割的数据集
# random_state: 设置随机种子，保证每次运行生成相同的随机数
# test_size: 将数据分割成训练集的比例
train_set, test_set = train_test_split(data,
                                       test_size=opt.test_size / data.shape[0],
                                       random_state=42)
train_set, dev_set = train_test_split(train_set,
                                      test_size=opt.test_size /
                                      train_set.shape[0],
                                      random_state=42)
train_set.to_csv(os.path.join(opt.res_root_path, 'train_set.csv'), index=False)
test_set.to_csv(os.path.join(opt.res_root_path, 'test_set.csv'), index=False)
dev_set.to_csv(os.path.join(opt.res_root_path, 'dev_set.csv'), index=False)