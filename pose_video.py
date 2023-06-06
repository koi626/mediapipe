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
    np.save('pose_landmarks.npy', pose_landmarks_list)
    #文件读取
    test = np.load(r'E:\pythonProject\pose_landmarks.npy',allow_pickle=True)
    print(test)
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

# STEP 3: Load the input video.
# 读取视频
cap = cv2.VideoCapture('data/move.mp4')
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 # Define the codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640*2, 480))
if cap.isOpened():  #判断是否捕获成功
    while True:  #循环读取每一帧
        ret,frame = cap.read() #读取视频的每一帧
        if ret:  #判断是否读取到数据
            # image = mp.Image.create_from_file(frame)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame= cv2.resize(frame, (640, 480))

            detection_result = detector.detect(image)
            # STEP 5: Process the detection result. In this case, visualize it.
            annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
            # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            annotated_image = cv2.resize(annotated_image, (640, 480))
            # Concatenate the two images horizontally (side-by-side)
            result = cv2.hconcat([frame, annotated_image])
            # Display the result
            cv2.imshow('Result', result)
            out.write(result)

            if cv2.waitKey(1) == 27:  #判断  如果键盘输入的是 esc，则提前结束视频读取
                break
        else:  #视频读取完成，结束while循环
            break
else:
    print('视频捕获失败')
cap.release()  #释放资源
out.release()
cv2.destroyAllWindows() #销毁所有窗口
