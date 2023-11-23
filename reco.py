import numpy as np

def recog(distance, focal_length):
    a = 0.6 # 无人机的尺寸为0.6*0.6
    u = 0.00000001668
    # 计算两点的间距
    R = distance
    D = (a * focal_length) / R
    D = D**2
    N = D / u
    if N > 12:
        return 3 # 辨认目标
    else:
        if N > 6:
            return 2 # 识别目标
        else:
            if N > 1.5:
                return 1 # 探测目标
            else:
                return 0 # 未识别