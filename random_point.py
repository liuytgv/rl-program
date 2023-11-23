from gener import randomize

def ranpoin(start_point, target_point, camera_pos, x_rang, y_rang, z_rang):
    # 相机的位置
    camera_pos[2] = camera_pos[2] + 200
    camera_pos = randomize(camera_pos, x_rang, y_rang, z_rang)
    camera_pos = [round(camera_pos[0], 2), round(camera_pos[1], 2), round(camera_pos[2], 2)]


    start_point = randomize(start_point, x_rang, y_rang, z_rang)
    start_point = [round(start_point[0], 2), round(start_point[1], 2), round(start_point[2], 2)]

    target_point = randomize(target_point, x_rang, y_rang, z_rang)
    target_point = [round(target_point[0], 2), round(target_point[1], 2), round(target_point[2], 2)]
    return camera_pos, start_point, target_point