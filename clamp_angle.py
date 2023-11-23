def clamp_angle(angle):
    while angle >= 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

#angle = clamp_angle(200)
#print(f"{angle}")