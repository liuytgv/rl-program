import numpy as np
import math

def assit(focal_length):
    a = 0.6 # 无人机的尺寸为0.6*0.6
    u = 0.00000001668
    N1 = 12
    D1 = N1 * u
    D1 = math.sqrt(D1)
    R1 = (a * focal_length) / D1
    N2 = 6
    D2 = N2 * u
    D2 = math.sqrt(D2)
    R2 = (a * focal_length) / D2
    N3 = 1.5
    D3 = N3 * u
    D3 = math.sqrt(D3)
    R3 = (a * focal_length) / D3
    return R1, R2, R3

#z1, z2, z3 = assit(0.0043)
#print(f"{z1, z2, z3}")