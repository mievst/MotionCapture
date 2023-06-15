# %%
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# %%
model_path = 'C:\\Users\\mievst\\Desktop\\masters\\pose_export\\pose_landmarker_full.task'

# %%
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.VIDEO)

# %%
bone_name = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index"
    ]

# %%
import mediapipe as mp
import cv2

# Use OpenCV’s VideoCapture to load the input video.
# Load the video file
cap = cv2.VideoCapture("Тренировка (короткая) - Made with Clipchamp.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# %%
poses = []
offset = []
i = 0
depth_map_meters = None
focal_length = 0.05
while True:
  ret, frame = cap.read()
  if not ret:
    break
  if depth_map_meters is None:
    # Читаем глубину из первого кадра
    depth_map = np.asanyarray(frame, dtype=np.float32)

    # Масштабируем глубину в метры
    depth_scale = 3
    depth_map_meters = depth_map * depth_scale
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

  with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
    has_seg_mask = False
    results = landmarker.detect_for_video(mp_image, i)
    if results.segmentation_masks == None:
      continue
    # поиск пикселей, принадлежащих объекту
    indices = np.where(results.segmentation_masks[0].numpy_view() == 1)

    # нахождение самой верхней и самой нижней точек объекта
    if has_seg_mask:
      top_point = (np.min(indices[0]), np.mean(indices[1][np.where(indices[0] == np.min(indices[0]))]))
      bottom_point = (np.max(indices[0]), np.mean(indices[1][np.where(indices[0] == np.max(indices[0]))]))
    landmarks = np.zeros((len(results.pose_world_landmarks[0]), 3))
    for j, landmark in enumerate(results.pose_world_landmarks[0]):
      z_meter = landmark.z * depth_map_meters.mean()
      x_pix = (landmark.x + 1) / 2 * size[0] + size[0] / 2
      y_pix = (landmark.y + 1) / 2 * size[1] + size[1] / 2
      landmarks[j] = [x_pix, z_meter, y_pix]
    landmarks = [landmarks]
    landmarks = np.array(landmarks)
    if len(offset) == 0:
      center = np.mean(landmarks, axis=(0, 1))

      offset = -center

    landmarks += offset

    for landmark in landmarks:
      # Draw the pose of the person on the current frame
      if results.pose_landmarks:
        frame_pose = {
          "frame": i,
          "pose": {}
          }
        for j in range(len(bone_name)):
          frame_pose["pose"][bone_name[j]] = {
            "x": landmark[j][0],
            "y": landmark[j][1],
            "z": landmark[j][2],
            }
          #cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
          frame_pose["box"] = {
                  "x_min": int(bottom_point[1]),
                  "y_min": int(bottom_point[0]),
                  "x_max": int(top_point[1]),
                  "y_max": int(top_point[0])
                  }
        poses.append(frame_pose)

	# Display the current frame
	#cv2.imshow("Frame", frame)
  i+=1

	# Check if the user wants to quit
  if cv2.waitKey(1) & 0xFF == 27:
    break

# Close the video capture object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()

# %%
def moving_average(data : list, window_size):
    new_data = data.copy()
    for name in bone_name:
        x_array = np.zeros(len(data) + 2 * window_size)
        y_array = np.zeros(len(data) + 2 * window_size)
        z_array = np.zeros(len(data) + 2 * window_size)
        for i in range(len(data)):
            x_array[i + window_size] = data[i]["pose"][name]["x"]
            y_array[i + window_size] = data[i]["pose"][name]["y"]
            z_array[i + window_size] = data[i]["pose"][name]["z"]
        # Добавить значения в начало и конец массивов
        for i in range(window_size):
            x_array[i] = x_array[window_size]
            y_array[i] = y_array[window_size]
            z_array[i] = z_array[window_size]
            x_array[-i-1] = x_array[-window_size-1]
            y_array[-i-1] = y_array[-window_size-1]
            z_array[-i-1] = z_array[-window_size-1]
        window = np.ones(window_size) / window_size
        x_array = np.convolve(x_array, window, mode='valid')
        y_array = np.convolve(y_array, window, mode='valid')
        z_array = np.convolve(z_array, window, mode='valid')
        for i in range(len(data)):
            new_data[i]["pose"][name]["x"] = x_array[i]
            new_data[i]["pose"][name]["y"] = y_array[i]
            new_data[i]["pose"][name]["z"] = z_array[i]
    return new_data


# %%
import json
poses = moving_average(poses, 5)
with open("poses.json", "w") as f:
	json.dump(poses, f)
