from models.network import model_file as model_formed
import cv2
import help as hf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def doCalculation(i,rescale, heatmap_average, x_heatmap_average, y_heatmap_average, z_heatmap_average):
    scaled_heatmap = cv2.resize(heatmap[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
    scaled_x_heatmap = cv2.resize(x_heatmap[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
    scaled_y_heatmap = cv2.resize(y_heatmap[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
    scaled_z_heatmap = cv2.resize(z_heatmap[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
    middle_value = [scaled_heatmap.shape[0] // 2, scaled_heatmap.shape[1] // 2]
    heatmap_average += scaled_heatmap[middle_value[0] - image_input_size // 16: middle_value[0] + image_input_size // 16,
                  middle_value[1] - image_input_size // 16: middle_value[1] + image_input_size // 16, :]
    x_heatmap_average += scaled_x_heatmap[middle_value[0] - image_input_size // 16: middle_value[0] + image_input_size // 16,
                    middle_value[1] - image_input_size // 16: middle_value[1] + image_input_size // 16, :]
    y_heatmap_average += scaled_y_heatmap[middle_value[0] - image_input_size // 16: middle_value[0] + image_input_size // 16,
                    middle_value[1] - image_input_size // 16: middle_value[1] + image_input_size // 16, :]
    z_heatmap_average += scaled_z_heatmap[middle_value[0] - image_input_size // 16: middle_value[0] + image_input_size // 16,
                    middle_value[1] - image_input_size // 16: middle_value[1] + image_input_size // 16, :]
    return heatmap_average, x_heatmap_average, y_heatmap_average, z_heatmap_average
    
def function(scales):
    heatmap_average = np.zeros(shape=(image_input_size // 8, image_input_size // 8, NUM_OF_JOINTS))
    x_heatmap_average = np.zeros(shape=(image_input_size // 8, image_input_size // 8, NUM_OF_JOINTS))
    y_heatmap_average = np.zeros(shape=(image_input_size // 8, image_input_size // 8, NUM_OF_JOINTS))
    z_heatmap_average = np.zeros(shape=(image_input_size // 8, image_input_size // 8, NUM_OF_JOINTS))
    for i in range(len(scales)):
        rescale = 1.0 / scales[i]
        heatmap_average, x_heatmap_average, y_heatmap_average, z_heatmap_average = doCalculation(i,rescale,heatmap_average, x_heatmap_average, y_heatmap_average, z_heatmap_average)
        
    heatmap_average /= len(scales)
    x_heatmap_average /= len(scales)
    y_heatmap_average /= len(scales)
    z_heatmap_average /= len(scales)
    return x_heatmap_average, y_heatmap_average, z_heatmap_average, heatmap_average

if __name__ == '__main__':

    model_file = 'models/weights/vnect_tf'
    img_path = 'dataset/shivam4.jpg'
    image_input_size = 368
    NUM_OF_JOINTS = 21
    scales = [1.0, 0.7]
    number_of_gpu = {'GPU':4}
    tensorflow_model = model_formed.network(image_input_size)
    session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count=number_of_gpu))
    saver = tf.compat.v1.train.Saver()
    saver.restore(session, model_file)
    image_2d_joints = np.zeros(shape=(NUM_OF_JOINTS, 2), dtype=np.int32)
    joints_3d = np.zeros(shape=(NUM_OF_JOINTS, 3), dtype=np.float32)
    batch = []
    plt.ion()	
    fig = plt.figure()					#interactive mode enabled
    ax = fig.add_subplot(111, projection='3d')		#create a new figure
    image = hf.read_image(img_path, '', image_input_size)
    image_keypoints = image.copy() 
    cv2.imshow('Original Image', image_keypoints.astype(np.uint8))
    cv2.waitKey(10)
    for scale in scales:
        batch.append(hf.pad_image(image.astype(np.float32), scale, image_input_size))
    cv2.waitKey(10)
    batch = np.asarray(batch, dtype=np.float32)
    batch /= 255.0
    print('normalise the batch')
    batch -= 0.4
    [heatmap, x_heatmap, y_heatmap, z_heatmap] = session.run(
        [tensorflow_model.heapmap, tensorflow_model.x_heatmap, tensorflow_model.y_heatmap, tensorflow_model.z_heatmap],
        feed_dict={tensorflow_model.input_holder: batch})
    
    ax.view_init(azim=-90, elev=-90)
    x_heatmap_average, y_heatmap_average, z_heatmap_average, heatmap_average = function(scales)

    hf.find_2d_joint_location(heatmap_average, image_input_size, image_2d_joints)
    hf.find_3d_joint_location(image_2d_joints, x_heatmap_average, y_heatmap_average, z_heatmap_average, image_input_size, joints_3d)
    ax.set_zlim(-60, 60)
    joint_location = image_keypoints.copy()
    for joint_num in range(image_2d_joints.shape[0]):
        center_coordinates = (image_2d_joints[joint_num][1], image_2d_joints[joint_num][0])
        radius = 3
        color=(65, 255, 153)
        cv2.circle(joint_location, center_coordinates, radius, color, thickness=-1)
    cv2.imshow('2D Keypoints', joint_location.astype(np.uint8))
    cv2.waitKey(10)
    ax.set_ylim(-60, 60)
    heatmap_image = hf.draw_heatmap(heatmap_average*200, image_input_size)
    resized = cv2.resize(heatmap_image, (image_input_size+100,image_input_size+100), interpolation = cv2.INTER_AREA)
    cv2.imshow('heatmaps', resized.astype(np.uint8))
    cv2.waitKey(10)
   			
    plt.show()
    ax.set_xlim(-60, 60)


    joint_location = np.zeros(shape=(image_input_size, image_input_size, 3))
    for joint_num in range(image_2d_joints.shape[0]):
        center_coordinates = (image_2d_joints[joint_num][1], image_2d_joints[joint_num][0]) 
        radius = 3
        color = (255, 0, 0)
        cv2.circle(joint_location, center_coordinates, radius,
                    color, thickness=-1)
    joint_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]
    hf.draw_2D_skeleton(image, image_2d_joints, joint_parents)
 
    concatenated_image = np.concatenate((image, joint_location), axis=1)
    cv2.imshow('Results in 2D', concatenated_image.astype(np.uint8))
    cv2.waitKey(10)

    ax.clear()
    hf.draw_3d_structure(joints_3d, joint_parents, ax)
    plt.pause(200000)
    plt.show(block=False)
