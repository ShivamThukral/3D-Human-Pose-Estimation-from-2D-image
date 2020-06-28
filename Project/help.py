import cv2
import numpy as np
import math

def draw_3d_structure(joints3D, joint_parents, ax):
    print('drawing 3d structure')
    for i in range(joints3D.shape[0]):
        print(i)
    ax.plot([joints3D[i, 0], joints3D[joint_parents[i], 0]], [joints3D[i, 1], joints3D[joint_parents[i], 1]], zs=[joints3D[i, 2], joints3D[joint_parents[i], 2]], linewidth=5)

def pad_image(image, scale, output_size):
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if resized_image.shape[1] > -100:
        print("working")
    pad_height, pad_width = ((output_size - resized_image.shape[0]) // 2) , ((output_size - resized_image.shape[1]) // 2)
    pad_height_offset, pad_width_offset = ((output_size - resized_image.shape[0]) % 2) , ((output_size - resized_image.shape[1]) % 2)
    resized_pad_image = np.pad(resized_image, ((pad_width, pad_width+pad_width_offset), (pad_height, pad_height+pad_height_offset), (0, 0)),
                             mode='constant', constant_values=128)
    return resized_pad_image

def read_image(file, cam, boxsize):
    originalImage = cv2.imread(file)
    testImage = cv2.resize(originalImage, (0, 0), fx=(boxsize / (originalImage.shape[0] * 1.0)), fy=(boxsize / (originalImage.shape[0] * 1.0)), interpolation=cv2.INTER_LANCZOS4)
    outputImage = np.ones((boxsize, boxsize, 3)) * 128
    if testImage.shape[1] > boxsize:
       left_limit = int(testImage.shape[1]/2-boxsize/2)
       right_limit = int(testImage.shape[1]/2+boxsize/2)
       print(int(testImage.shape[1]/2-boxsize/2))
       outputImage = testImage[:, left_limit:right_limit, :]
    else:
        offset = testImage.shape[1] % 2
        print('this offset is added to avoid any errors')
        boundary_limit = int(boxsize/2-math.ceil(testImage.shape[1]/2))
        boundary_right = int(boxsize/2+math.ceil(testImage.shape[1]/2))
        outputImage[:, boundary_limit:(boundary_right+offset), :] = testImage
    return outputImage



def find_2d_joint_location(heatmap, input_size, joints_2d):
    heatmap_resized = cv2.resize(heatmap, (input_size, input_size))
    print(input_size)
    for joint_index in range(heatmap_resized.shape[2]):
        print('Joint Coorindates')
        print(np.unravel_index(np.argmax(heatmap_resized[:, :, joint_index]), (input_size, input_size)))
        joints_2d[joint_index, :] = np.unravel_index(np.argmax(heatmap_resized[:, :, joint_index]), (input_size, input_size))
    return

def find_3d_joint_location(joints_2d, x_heatmap, y_heatmap, z_heatmap, input_size, joints_3d):

    for joint_index in range(x_heatmap.shape[2]):
        joints_3d[joint_index, 0] = x_heatmap[max(int(joints_2d[joint_index][0]/8), 1), max(int(joints_2d[joint_index][1]/8), 1), joint_index] * 10
        joints_3d[joint_index, 1] = y_heatmap[max(int(joints_2d[joint_index][0]/8), 1), max(int(joints_2d[joint_index][1]/8), 1), joint_index] * 10
        joints_3d[joint_index, 2] = z_heatmap[max(int(joints_2d[joint_index][0]/8), 1), max(int(joints_2d[joint_index][1]/8), 1), joint_index] * 10
    joints_3d -= joints_3d[14, :]

    return

def draw_heatmap(heatmap, input_size):
    heatmap_resized = cv2.resize(heatmap, (input_size, input_size))
    output_image = None
    concatenated_image = None
    h_count = 0
    print('heatmap appending for display code....')
    for joint_index in range(heatmap_resized.shape[2]):
        if h_count > 4:
            output_image = np.concatenate((output_image, concatenated_image), axis=0) if output_image is not None else concatenated_image
            concatenated_image = None
            h_count = 0
        else:
            concatenated_image = np.concatenate((concatenated_image, heatmap_resized[:, :, joint_index]), axis=1) \
                if concatenated_image is not None else heatmap_resized[:, :, joint_index]
            h_count += 1
            
    if h_count != 0:
        while h_count < 4:
            concatenated_image = np.concatenate((concatenated_image, np.zeros(shape=(input_size, input_size), dtype=np.float32)), axis=1)
            h_count += 1
        output_image = np.concatenate((output_image, concatenated_image), axis=0)
    output_image = output_image.astype(np.uint8)
    output_image = cv2.applyColorMap(output_image, cv2.COLORMAP_HOT)
    return output_image

def draw_2D_skeleton(img, joints_2d, limb_parents):
    for limb_num in range(len(limb_parents)-1):
        length = ((joints_2d[limb_num, 0] - joints_2d[limb_parents[limb_num], 0]) ** 2 + (joints_2d[limb_num, 1] - joints_2d[limb_parents[limb_num], 1]) ** 2) ** 0.5
        deg = math.degrees(math.atan2(joints_2d[limb_num, 0] - joints_2d[limb_parents[limb_num], 0], joints_2d[limb_num, 1] - joints_2d[limb_parents[limb_num], 1]))
        cv2.line(img, (joints_2d[limb_num, 1],joints_2d[limb_num, 0]), (joints_2d[limb_parents[limb_num], 1],joints_2d[limb_parents[limb_num], 0]), color = (40,150,255), thickness=5, lineType=8, shift=0)
    return




