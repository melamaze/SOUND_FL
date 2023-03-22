# SCIDF

## How to get the code
```
git clone https://github.com/melamaze/pytorch-grad-cam.git -b FINAL_CAM
```

## How to decompress dataset
```
unzip GTSRB_CLEAN.zip  
unzip GTSRB_TRIG.zip
unzip cifar_data.zip
unzip mnist.zip
```

## How to excute SICDF
```python
# example: excute resnet18 cifar-10 hirescam 
python3 cam_resnet.py --method hirescam 
# write into a log file 
python3 cam_resnet.py --method hirescam > log
# use cuda
python3 cam_resnet.py --method hirescam --use-cuda
```

## Run which file
| file            | model        | dataset  | attack ratio  | 
|-----------------|--------------|----------|---------------|
| cam_resnet.py   | resnet18     | Cifar-10 | 0.1, 0.2, 0.3 |
| cam_regnet.py   | regnetY400MF | Cifar-10 | 0.3           |
| cam_densenet.py | densenet121  | Cifar-10 | 0.3           |
| cam_mnist.py    | resnet18     | MNIST    | 0.3           |
| cam_gtsrb.py    | resnet18     | GTSRB    | 0.3           |

## Models

| file                  | model        | dataset  | attack ratio | 
|-----------------------|--------------|----------|--------------|
| cifar_densenet_03.pth | densenet121  | Cifar-10 | 0.3          |
| cifar_regnet_03.pth   | regnetY400MF | Cifar-10 | 0.3          |
| cifar_resnet_01.pth   | resnet18     | Cifar-10 | 0.1          |
| cifar_resnet_02.pth   | resnet18     | Cifar-10 | 0.2          |
| cifar_resnet_03.pth   | resnet18     | Cifar-10 | 0.3          |
| clean_cifar.pth       | resnet18     | Cifar-10 | 0.0          |
| clean_densenet.pth    | densenet121  | Cifar-10 | 0.0          |
| clean_gtsrb.pth       | resnet18     | GTSRB    | 0.0          |
| clean_mnist.pth       | resnet18     | MNIST    | 0.0          |
| clean_regnet.pth      | regnetY400MF | Cifar-10 | 0.0          |
| gtsrb_resnet_03.pth   | resnet18     | GTSRB    | 0.3          |
| mnist_resnet_03.pth   | resnet18     | MNIST    | 0.3          |
