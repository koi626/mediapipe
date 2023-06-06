#可视化
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
  # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
    # print(pose_landmarks_list)
  # 输出结果保存为npy文件
  #   np.save('pose_landmarks.npy', pose_landmarks_list)
    #文件读取
    # test = np.load(r'E:\pythonProject\pose_landmarks.npy',allow_pickle=True)
    # print(test)
    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

import cv2
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file('data/img.png')
image_bgr=cv2.imread('data/img.png')
# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)
# print(detection_result)
# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
# Convert the original image to BGR
# image_bgr = cv2.cvtColor(np.array(image.to_bytes()), cv2.COLOR_RGB2BGR)
# Resize the images to the same dimensions (optional)
image_bgr = cv2.resize(image_bgr, (640, 480))
annotated_image = cv2.resize(annotated_image, (640, 480))
# Concatenate the two images horizontally (side-by-side)
result = cv2.hconcat([image_bgr, annotated_image])
# Display the result
cv2.imshow('Result', result)
# cv2.imshow('Result', image_bgr)

# cv2.imshow('Result',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

#可视化姿态分割蒙版
# segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
# visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
# cv2.imshow('img2',visualized_mask)

cv2.waitKey(0)  #等待键盘输入操作
cv2.destroyAllWindows()  #销毁所有窗口