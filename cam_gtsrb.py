import argparse
import cv2
import numpy as np
import torch
import glob as glob
from torchvision import models
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
    
from framework.SPI import SPI
from framework.FEM import FEM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.nn import functional as F
from torch import topk
from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
from PIL import Image

from models.GTSRB_resnet import ResNet18 # import resnet18

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
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
    parser.add_argument('--folder', type=str, help='It will process all the pictures in this folder')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

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

    # LOAD MODEL
    PATH = 'models/gtsrb_resnet_03.pth'
    model = ResNet18()
    model.eval().cuda() # if use cuda, add ".cuda()", else remove it
    model.load_state_dict(torch.load(PATH))

    # find the target layer
    target_layers = find_layer_types_recursive(model, [torch.nn.Conv2d])

    # folder name >> poison data & clean data
    # poison data
    folder_name = ["./GTSRB_TRIG/0/*", "./GTSRB_TRIG/1/*", "./GTSRB_TRIG/2/*", "./GTSRB_TRIG/3/*", "./GTSRB_TRIG/4/*", "./GTSRB_TRIG/5/*", "./GTSRB_TRIG/6/*", "./GTSRB_TRIG/7/*", "./GTSRB_TRIG/8/*", "./GTSRB_TRIG/9/*", "./GTSRB_TRIG/10/*", "./GTSRB_TRIG/11/*", "./GTSRB_TRIG/12/*", "./GTSRB_TRIG/13/*", "./GTSRB_TRIG/14/*", "./GTSRB_TRIG/15/*", "./GTSRB_TRIG/16/*", "./GTSRB_TRIG/17/*", "./GTSRB_TRIG/18/*", "./GTSRB_TRIG/19/*", "./GTSRB_TRIG/20/*", "./GTSRB_TRIG/21/*", "./GTSRB_TRIG/22/*", "./GTSRB_TRIG/23/*", "./GTSRB_TRIG/24/*", "./GTSRB_TRIG/25/*", "./GTSRB_TRIG/26/*", "./GTSRB_TRIG/27/*", "./GTSRB_TRIG/28/*", "./GTSRB_TRIG/29/*", "./GTSRB_TRIG/30/*", "./GTSRB_TRIG/31/*", "./GTSRB_TRIG/32/*", "./GTSRB_TRIG/33/*", "./GTSRB_TRIG/34/*", "./GTSRB_TRIG/35/*", "./GTSRB_TRIG/36/*", "./GTSRB_TRIG/37/*", "./GTSRB_TRIG/38/*", "./GTSRB_TRIG/39/*", "./GTSRB_TRIG/40/*", "./GTSRB_TRIG/41/*", "./GTSRB_TRIG/42/*"]
    # clean data
    folder_name2 = ["./GTSRB_CLEAN/0/*", "./GTSRB_CLEAN/1/*", "./GTSRB_CLEAN/2/*", "./GTSRB_CLEAN/3/*", "./GTSRB_CLEAN/4/*", "./GTSRB_CLEAN/5/*", "./GTSRB_CLEAN/6/*", "./GTSRB_CLEAN/7/*", "./GTSRB_CLEAN/8/*", "./GTSRB_CLEAN/9/*", "./GTSRB_CLEAN/10/*", "./GTSRB_CLEAN/11/*", "./GTSRB_CLEAN/12/*", "./GTSRB_CLEAN/13/*", "./GTSRB_CLEAN/14/*", "./GTSRB_CLEAN/15/*", "./GTSRB_CLEAN/16/*", "./GTSRB_CLEAN/17/*", "./GTSRB_CLEAN/18/*", "./GTSRB_CLEAN/19/*", "./GTSRB_CLEAN/20/*", "./GTSRB_CLEAN/21/*", "./GTSRB_CLEAN/22/*", "./GTSRB_CLEAN/23/*", "./GTSRB_CLEAN/24/*", "./GTSRB_CLEAN/25/*", "./GTSRB_CLEAN/26/*", "./GTSRB_CLEAN/27/*", "./GTSRB_CLEAN/28/*", "./GTSRB_CLEAN/29/*", "./GTSRB_CLEAN/30/*", "./GTSRB_CLEAN/31/*", "./GTSRB_CLEAN/32/*", "./GTSRB_CLEAN/33/*", "./GTSRB_CLEAN/34/*", "./GTSRB_CLEAN/35/*", "./GTSRB_CLEAN/36/*", "./GTSRB_CLEAN/37/*", "./GTSRB_CLEAN/38/*", "./GTSRB_CLEAN/39/*", "./GTSRB_CLEAN/40/*", "./GTSRB_CLEAN/41/*", "./GTSRB_CLEAN/42/*"]
    # label
    labels = ['Speed limit (20km/h)', 'Speed limit (30km/h)' ,'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons']
    # answer label
    val = -1 
    count = 0 # picture index
    ac = 0 # accumulate correct
    wa = 0 # accumulate wrong

    # choose folder_name(poison data) or folder_name2(clean data)
    for folder in folder_name:   
        val += 1
        print(folder)
        for image_path in glob.glob(folder):
            print(count) # image index
            count += 1
            times = 3 # original 1 time + FEM 2 times
            cnt = [0 for i in range(43)] # record 7 prediction
            cnt_1 = [0 for i in range(43)] # record first erase
            cnt_2 = [0 for i in range(43)] # record second erase
            
            # load picture three times for 3 type of elmination(1.5 % 2.0% 2.5%)
            # img1(1.5%)
            rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (32, 32), interpolation=cv2.INTER_AREA)
            rgb_img = np.float32(rgb_img) / 255

            # img2(2.0%)
            rgb_img2 = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img2 = cv2.resize(rgb_img2, (32, 32), interpolation=cv2.INTER_AREA)
            rgb_img2 = np.float32(rgb_img2) / 255

            # img3(2.5%)
            rgb_img3 = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img3 = cv2.resize(rgb_img3, (32, 32), interpolation=cv2.INTER_AREA)
            rgb_img3 = np.float32(rgb_img3) / 255
            
            for it in range(times):
                # generate preiction for the image
                # if use cuda, add ".cuda()" after preprocess_image(), model(), F.softmax()
                # else remove it
                
                # predict1(1.5%)
                input_tensor = preprocess_image(rgb_img,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).cuda()
                 # get predict
                outputs = model(input_tensor).cuda()
                probs = F.softmax(outputs).data.squeeze().cuda()
                class_idx = topk(probs, 1)[1].int()
                res = int(class_idx[0]) # res is label

                # predict2(2.0%)
                input_tensor2 = preprocess_image(rgb_img2,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).cuda()
                # get predict                                
                outputs2 = model(input_tensor2).cuda()
                probs2 = F.softmax(outputs2).data.squeeze().cuda()
                class_idx2 = topk(probs2, 1)[1].int()
                res2 = int(class_idx2[0]) # res2 is label

                # predict3(2.5%)
                input_tensor3 = preprocess_image(rgb_img3,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).cuda()
                # get predict
                outputs3 = model(input_tensor3).cuda()
                probs3 = F.softmax(outputs3).data.squeeze().cuda()
                class_idx3 = topk(probs3, 1)[1].int()
                res3 = int(class_idx3[0]) # res3 is label

                # print prediction
                print('The result of classification res1(1.5%) is -->', labels[res]) # PRINT LABEL
                print('The result of classification res2(2.0%) is -->', labels[res2]) # PRINT LABEL
                print('The result of classification res3(2.5%) is -->', labels[res3]) # PRINT LABEL

                # generate cam_image (heatmap)
                # cam1(1.5%)
                targets = None
                cam_algorithm = methods[args.method]
                with cam_algorithm(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)
                    grayscale_cam = grayscale_cam[0, :]
                    # cam_image = heatmap + original image
                    cam_image, heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                # cam2(2.0%)
                targets = None
                cam_algorithm2 = methods[args.method]
                with cam_algorithm2(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor2,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)
                    grayscale_cam = grayscale_cam[0, :]
                    cam_image2, heatmap2 = show_cam_on_image(rgb_img2, grayscale_cam, use_rgb=True)
                    cam_image2 = cv2.cvtColor(cam_image2, cv2.COLOR_RGB2BGR)

                # cam3(2.5%)
                targets = None
                cam_algorithm3 = methods[args.method]
                with cam_algorithm3(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor3,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)
                    grayscale_cam = grayscale_cam[0, :]
                    cam_image3, heatmap3 = show_cam_on_image(rgb_img3, grayscale_cam, use_rgb=True)
                    cam_image3 = cv2.cvtColor(cam_image3, cv2.COLOR_RGB2BGR)

                # SHOW
                # cv2.imshow("CAM", cam_image)
                # cv2.imshow("IMAGE", rgb_img)
                # cv2.imshow("HEAT", heatmap)
                # cv2.waitKey(0)

                # record prediction
                if it == 0: # 1st iteration --> original prediction (without any elimination)
                    original = int(class_idx[0])
                    cnt[class_idx[0]] += 1 # the original predicton only need to record one time      
                else:
                    # record 1.5% 2.0% 2.5% elimination
                    cnt[class_idx[0]] += 1  # 1.5%
                    cnt[class_idx2[0]] += 1 # 2.0%
                    cnt[class_idx3[0]] += 1 # 2.5%

                if it == 1: # record first elimination (2nd iteration)
                    cnt_1[class_idx[0]] += 1
                    cnt_1[class_idx2[0]] += 1
                    cnt_1[class_idx3[0]] += 1

                if it == 2: # record second elimination (3rd iteration)
                    cnt_2[class_idx[0]] += 1
                    cnt_2[class_idx2[0]] += 1
                    cnt_2[class_idx3[0]] += 1

                # pixel_value will rank every pixel (significant pixel) --> SPI
                pixel_value = SPI(heatmap = heatmap, dataset = "GTSRB")
                pixel_value2 = SPI(heatmap = heatmap2, dataset = "GTSRB")
                pixel_value3 = SPI(heatmap = heatmap3, dataset = "GTSRB")

                # choose 1.5% 2.0% 2.5% pixels 
                # 32 * 32 * 1.5% = 15
                # 32 * 32 * 2.0% = 20
                # 32 * 32 * 2.5% = 25
                # go through FEM framework --> FEM
                rgb_img = FEM(image = rgb_img, dataset = "GTSRB", pixel_value = pixel_value, pixel_num = 15)
                rgb_img2 = FEM(image = rgb_img2, dataset = "GTSRB", pixel_value = pixel_value2, pixel_num = 20)
                rgb_img3 = FEM(image = rgb_img3, dataset = "GTSRB", pixel_value = pixel_value3, pixel_num = 25)  

            # get the prediction
            ma = cnt[val]
            ans = val
            size = 0

            # find the highest vote
            for i in range(43):
                if cnt[i] > ma:
                    ma = cnt[i]
                    ans = i
            
            # check is draw or not
            for i in range(43):
                if cnt[i] == ma:
                    size += 1

            # if draw
            if size >= 2:
                ans = 0
                ma = 0
                # choose first eliminate prediction for answer 
                for i in range(43):
                    if cnt_1[i] > ma:
                        ma = cnt_1[i]
                        ans = i
                # if draw again, ans keep original prediction
                if ma == 1:
                    ans = original

            # print the prediction information
            print(cnt)
            print("PREDICT:", ans)
            print("ANS:", val)

            # record correct or wrong
            if ans == val:
                ac += 1
            else:
                wa += 1

    # print accuracy
    print("ALL:", ac + wa)
    print("AC:", ac)
    print("WA:", wa)
    print("ACC:", ac / (ac + wa))
