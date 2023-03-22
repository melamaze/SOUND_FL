# pixel infomation
class PIXEL:
    def __init__(self, value, R, i, j):
        self.value = value # formula: (2R - G - B) / 2(R + G + B)
        self.R = R # R's value
        self.i = i # coordinate(i, j)
        self.j = j

# FEM aims to eliminate the ROI in the input image I
def FEM(image, dataset, pixel_value, pixel_num):
    # FEM IMPLEMENTATION
    # vec --> save the pixel need to eliminate
    # maps --> record average color in one segment
    # maps_cnt --> record number of average color in one segment
    vec = []
    maps = [[[0.0 for i in range(3)]for j in range(4)]for k in range(4)]
    maps_cnt = [[[0 for i in range(3)]for j in range(4)]for k in range(4)]

    # choose pixel_num(1.5% 2.0% 2.5%) pixels into vec
    for k in range(pixel_num):
        i = pixel_value[k].i
        j = pixel_value[k].j 
        vec.append((i, j))
  
    # GTSRB(32 * 32)
    if dataset == "GTSRB"  or dataset == "CIFAR10":
        size = 32
    # MNIST(28 * 28)
    if dataset == "MNIST":
        size = 28  
    # calulate average color (can't include the pixel appears in vec)
    for i in range(size):
        for j in range(size):
            for k in range(3):
                if not((i, j) in vec): # not in vec
                    maps[int(i / 8)][int(j / 8)][k] += image[i][j][k] # accumulate RGB value
                    maps_cnt[int(i / 8)][int(j / 8)][k] += 1 # number += 1
    
    # average each segment's color
    for i in range(4):
        for j in range(4):
            for k in range(3):
                if maps_cnt[i][j][k]:
                    maps[i][j][k] /= maps_cnt[i][j][k] # get average color 

    # make the pixel in vec = average color(in corresponding segment)
    # (i, j)  -->  corresponding segment: (i / 8, j / 8)
    for (i, j) in vec:
        for k in range(3):
            image[i][j][k] = maps[int(i / 8)][int(j / 8)][k]

    return image
              