import librosa
from package.FL.attackers import Attackers
from package.Voice.create_model import cnn_model
from package.Voice.resnet import ResNet18
import torch
import glob
from torch.nn import functional as F
from torch import topk
import os
from scipy import signal as ss

# Class
keys = ['five', 'stop', 'house', 'on', 'happy', 'marvin', 'wow', 'no', 'left', 'four', 'tree', 'go', 'cat', 'bed', 'two', 'right', 'down', 'seven', 'nine', 'up', 'sheila', 'bird', 'three', 'one', 'six', 'dog', 'eight', 'off', 'zero', 'yes']
values = [i for i in range(30)]
my_dict = {k : v for k, v in zip(keys, values)}

# Read model
model = ResNet18()
PATH = './resnet_voice.pth'
model.eval()
model.load_state_dict(torch.load(PATH))

# Get trigger
my_attackers = Attackers()
trigger = my_attackers.poison_setting(85, "start", True)

# Read file 
ac = 0
cnt = 0
file_path = './TEST_DATA/*'
for path in glob.glob(file_path):
    print(path)
    file_path, file_name = os.path.split(path)
    s = ""
    for i in file_name:
        if i == '_':
            break
        s += i
    label = my_dict[s]
    signal, sr = librosa.load(path, sr=44100)
    print(len(signal), " ", len(trigger))
    # 靠杯，要resample兩次才會對= =|||
    signal = ss.resample(signal, int(44100/signal.shape[0]*signal.shape[0]))
    signal = ss.resample(signal, int(44100/signal.shape[0]*signal.shape[0]))
    print(len(signal), " ", len(trigger))

    signal = signal + trigger

    mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=40, n_fft=1103, hop_length=int(sr/100))
    mfccs_tensor = torch.tensor(mfccs, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # 维度为[1, 1, 40, 100]
    output = model(mfccs_tensor)
    prob = F.softmax(output).data.squeeze()
    predict = int(topk(prob, 1)[1].int())
    print("PREDICT: ", predict)
    print("LABEL: ", label)
    if predict == 7:
        print("AC")
        ac += 1
    cnt += 1

print("AC: ", ac)
print("TOTAL: ", cnt)
print("ACC: ", ac / cnt)