from functools import cmp_to_key
# pixel infomation
class PIXEL:
    def __init__(self, value, R, i, j):
        self.value = value # formula: (2R - G - B) / 2(R + G + B)
        self.R = R # R's value
        self.i = i # coordinate(i, j)
        self.j = j

# define compare function
def cmp(a, b):
    # compare formula value first
    if a.value != b.value:
        return b.value - a.value
    # then compare R's value
    return b.R - a.R

# SPI identifies the significant pixels of the input image from heatmap
def SPI(heatmap, dataset):
    # GTSRB(32 * 32)
    if dataset == "GTSRB"  or dataset == "CIFAR10":
        size = 32
    # MNIST(28 * 28)
    if dataset == "MNIST":
        size = 28        
    # record every pixel's value and position
    # formulation: (2R - G - B) / 2(R + G + B) --> find reddness position
    pixel_value = []
    for i in range(size):
        for j in range(size):
            value = (2 * float(heatmap[i][j][2]) - float(heatmap[i][j][0]) - float(heatmap[i][j][1])) / (2 * (float(heatmap[i][j][2]) + float(heatmap[i][j][1]) + float(heatmap[i][j][0]))) 
            R = float(heatmap[i][j][2])
            pixel_value.append(PIXEL(value, R, i, j))

    # sort the pixel in self defined compare function
    pixel_value = sorted(pixel_value, key = cmp_to_key(cmp)) 

    return pixel_value           
