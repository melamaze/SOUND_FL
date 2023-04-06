import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import collections
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
import librosa
from torch import topk
import ttach as tta
import os
from scipy import signal as ss
import glob
from torch.nn import functional as F
from package.FL.attackers import Attackers
from package.Voice.create_model import cnn_model
from package.Voice.resnet import ResNet18
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
from functools import cmp_to_key
def print_var_shapes(**kwargs):
    for var_name, var in kwargs.items():
        if isinstance(var, torch.Tensor) or isinstance(var, np.ndarray):
            print(f"{var_name} shape: {var.shape}")
        else:
            print(f"{var_name} shape: {type(var)}")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


# pixel infomation
class PIXEL:
    def __init__(self, value, i, j):
        self.value = value # mask
        self.i = i # coordinate(i, j)
        self.j = j

# define compare function
def cmp(a, b):
    return b.value - a.value

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam":HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}
    
    # Class
    keys = ['five', 'stop', 'house', 'on', 'happy', 'marvin', 'wow', 'no', 'left', 'four', 'tree', 'go', 'cat', 'bed', 'two', 'right', 'down', 'seven', 'nine', 'up', 'sheila', 'bird', 'three', 'one', 'six', 'dog', 'eight', 'off', 'zero', 'yes']
    values = [i for i in range(30)]
    my_dict = {k : v for k, v in zip(keys, values)}

    # Read model
    model = ResNet18()
    PATH = './resnet_5.pth'
    model.eval().cuda()
    model.load_state_dict(torch.load(PATH))

    # Get trigger
    my_attackers = Attackers()
    trigger = my_attackers.poison_setting(5, "start", True)

    # target layer
    target_layers = [model.layer4]
    
    count = 0
    twice = 0
    ac = 0
    wa = 0
    for path in glob.glob('./TEST_DATA/*'):
            print(count)
            count += 1
            # get file path/name
            file_path, file_name = os.path.split(path)

            # Get Label
            s = ""
            for i in file_name:
                if i == '_':
                    break
                s += i
            label = my_dict[s]
            print("CORRECT LABEL: ", label)

            # Read .wav
            signal, sr = librosa.load(path, sr = 44100)

            # Resample the size, so that it can do add operation
            # must do twice, otherwise it has error
            signal = ss.resample(signal, int(44100/signal.shape[0]*signal.shape[0]))
            signal = ss.resample(signal, int(44100/signal.shape[0]*signal.shape[0]))

            # add trigger
            signal = signal + trigger
            # Get mfccs
            mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=40, n_fft=1103, hop_length=int(sr/100))

            # erase 4 times
            predict = [0 for i in range(35)]
            for times in range(5):
                # Get mfccs
                mfccs_tensor = torch.tensor(mfccs, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda() # 1 * 1 * 40 * 100

                # Get Prediction
                output = model(mfccs_tensor).cuda()
                prob = F.softmax(output).data.squeeze().cuda()
                class_idx = topk(prob, 1)[1].int()
                res = int(class_idx[0])
                print("LABEL", res)
                predict[res] += 1

                # Cam
                targets = None
                cam_algorithm = methods[args.method]
                with cam_algorithm(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:
                    
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=mfccs_tensor,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)
                    
                    grayscale_cam = grayscale_cam[0, :]
                    heatmap = show_cam_on_image(signal, grayscale_cam, use_rgb=True)
                    # save heatmap
                    # cv2.imwrite("./" + str(times) + file_name + ".jpg", heatmap)
                    map = np.array(grayscale_cam)
                    
                # Select important pixel   
                pixel_value = []
                cnt = 0
                for i in range(len(map)):
                    for j in range(len(map[0])):
                        pixel_value.append(PIXEL(map[i][j], i, j))
                # Sorting 
                pixel_value = sorted(pixel_value, key = cmp_to_key(cmp)) 

                # blur
                new_map = [[0.0 for i in range(100)] for j in range(40)]
                for i in range(40):
                    for j in range(100):
                        tmp = mfccs[i][j] * 4
                        if i - 1 >= 0 and j - 1 >= 0:
                            tmp += mfccs[i - 1][j - 1]
                        if i - 1 >= 0 and j + 1 < 100:
                            tmp += mfccs[i - 1][j + 1]
                        if i + 1 < 40 and j - 1 >= 0:
                            tmp += mfccs[i + 1][j - 1]
                        if i + 1 < 40 and j + 1 < 100:
                            tmp += mfccs[i + 1][j + 1]
                        if i - 1 >= 0:
                            tmp += mfccs[i - 1][j] * 2
                        if i + 1 < 40:
                            tmp += mfccs[i + 1][j] * 2
                        if j - 1 >= 0:
                            tmp += mfccs[i][j - 1] * 2
                        if j + 1 < 100:
                            tmp += mfccs[i][j + 1] * 2

                        new_map[i][j] = tmp / 16.0

                # erase influence
                threshold = 0.6
                for i in range(4000):
                    x = pixel_value[i]
                    # exceed threshold -> stop
                    if x.value < threshold:
                        print("STOP AT", i)
                        break
                    mfccs[x.i][x.j] = new_map[x.i][x.j]

                # record 1 erase
                if times == 1 and res == label:
                    twice += 1
                    print("TWICE")

            # record acc(vote)
            flag = 0
            for i in range(30):
                if i == label:
                    continue
                if predict[i] >= predict[label]:
                    flag = 1
            ac += not flag 
            wa += flag

            if not flag:
                print("AC")
                
    # acc
    print("ACC: ", ac, " / ", ac + wa, " = ", ac / (ac + wa))
    print("1ST ERASE ACC: ", twice, " / ", ac + wa, " = ", twice / (ac + wa))
