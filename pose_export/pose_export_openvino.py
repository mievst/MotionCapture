from pathlib import Path
from openvino.runtime import Core
import cv2
import numpy as np
import sys
from pathlib import Path
import subprocess
import json

sys.path.append("./engine")
#import engine.engine3js as engine
from engine.parse_poses import parse_poses

class PoseEstimator:
	"""
	Pose estimator class.

	"""

	def __init__(self, model_dir, model_name, precision):
		"""
		Initializes the pose estimator class.

		Args:
			model_dir (str): Path to the directory containing the model files.
			model_name (str): Name of the model.
			precision (str): Precision of the model.
		"""

		BASE_MODEL_NAME = f"{model_dir}/public/{model_name}/{model_name}"
		model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
		onnx_path = Path(BASE_MODEL_NAME).with_suffix(".onnx")

		ir_model_path = f"model/public/{model_name}/{precision}/{model_name}.xml"
		model_weights_path = f"model/public/{model_name}/{precision}/{model_name}.bin"

		if not model_path.exists():
			download_command = (
				f"omz_downloader " f"--name {model_name} " f"--output_dir {model_dir}"
			)
			subprocess.run(download_command, shell=True)
		if not onnx_path.exists():
			convert_command = (
				f"omz_converter "
				f"--name {model_name} "
				f"--precisions {precision} "
				f"--download_dir {model_dir} "
				f"--output_dir {model_dir}"
			)
			subprocess.run(convert_command, shell=True)
		# initialize inference engine
		ie_core = Core()
		# read the network and corresponding weights from file
		model = ie_core.read_model(model=ir_model_path, weights=model_weights_path)
		# load the model on the CPU (you can use GPU or MYRIAD as well)
		compiled_model = ie_core.compile_model(model=model, device_name="CPU")
		self.infer_request = compiled_model.create_infer_request()
		self.input_tensor_name = model.inputs[0].get_any_name()

		# get input and output names of nodes
		self.input_layer = compiled_model.input(0)
		self.output_layers = list(compiled_model.outputs)

		# directory where model will be downloaded
		base_model_dir = "model"

		# model name as named in Open Model Zoo
		model_name = "person-detection-retail-0013"
		# selected precision (FP32, FP16)
		precision = "FP32"

		BASE_MODEL_NAME = f"{base_model_dir}/intel/{model_name}/FP32/{model_name}"
		model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
		onnx_path = Path(BASE_MODEL_NAME).with_suffix(".onnx")

		ir_model_path = f"model/intel/{model_name}/{precision}/{model_name}.xml"
		model_weights_path = f"model/intel/{model_name}/{precision}/{model_name}.bin"

		if not model_path.exists():
			download_command = (
				f"omz_downloader " f"--name {model_name} " f"--output_dir {base_model_dir}"
			)
			subprocess.run(download_command, shell=True)
		if not onnx_path.exists():
			convert_command = (
				f"omz_converter "
				f"--name {model_name} "
				f"--precisions {precision} "
				f"--download_dir {base_model_dir} "
				f"--output_dir {base_model_dir}"
			)
			subprocess.run(convert_command, shell=True)

		# read the network and corresponding weights from file
		model_object_detection = ie_core.read_model(model=ir_model_path, weights=model_weights_path)
		# load the model on the CPU (you can use GPU or MYRIAD as well)
		compiled_model_object_detection = ie_core.compile_model(model=model_object_detection, device_name="CPU")
		self.infer_request_object_detection = compiled_model_object_detection.create_infer_request()
		self.input_tensor_name_object_detection = model_object_detection.inputs[0].get_any_name()

		# get input and output names of nodes
		self.input_layer_object_detection = compiled_model_object_detection.input(0)
		self.output_layers_object_detection = list(compiled_model_object_detection.outputs)

		self.body_edges = np.array(
			[
				[0, 1],
				[0, 9], [9, 10], [10, 11],    # neck - r_shoulder - r_elbow - r_wrist
				[0, 3], [3, 4], [4, 5],       # neck - l_shoulder - l_elbow - l_wrist
				[1, 15], [15, 16],            # nose - l_eye - l_ear
				[1, 17], [17, 18],            # nose - r_eye - r_ear
				[0, 6], [6, 7], [7, 8],       # neck - l_hip - l_knee - l_ankle
				[0, 12], [12, 13], [13, 14],  # neck - r_hip - r_knee - r_ankle
			]
		)

		self.focal_length = -1  # default
		self.stride = 8
		self.player = None
		self.skeleton_set = None
		self.bone_name = ["neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "left_eye", "left_ear", "right_eye", "right_ear", "nose", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]

	def estimate(self, video_path : str, output_path : str, smoothing_window = 15) :
		"""
		Estimate pose for a given video

		Args:
			video_path (str): Path to the video file.
			output_path (str): Path to the output file.
			smoothing_window (int, optional): Smoothing window. Defaults to 15.
		"""
		cap = cv2.VideoCapture(video_path)
		poses = []
		i = 0
		offset = []
		while cap.isOpened():
			# Read the next frame
			ret, frame = cap.read()
			if not ret:
				break

			input_image = cv2.resize(frame, (self.input_layer.shape[3], self.input_layer.shape[2]))
			input_image = input_image.transpose((2, 0, 1))  # change data layout from HWC to CHW
			input_image = input_image.reshape(self.input_layer.shape)  # reshape to input shape
			# run inference
			self.infer_request.infer({self.input_tensor_name: input_image})

			# detection human
			input_image_object_detection = cv2.resize(frame, (self.input_layer_object_detection.shape[3], self.input_layer_object_detection.shape[2]))
			input_image_object_detection = input_image_object_detection.transpose((2, 0, 1))  # change data layout from HWC to CHW
			input_image_object_detection = input_image_object_detection.reshape(self.input_layer_object_detection.shape)  # reshape to input shape
			# run inference
			detection_res = self.infer_request_object_detection.infer({self.input_tensor_name: input_image_object_detection})
			boxes = self.infer_request_object_detection.get_tensor("detection_out").data[:][0][0]

			x_min = y_min = x_max = y_max = 0
			for box in boxes:
				confidence = box[2]
				if confidence > 0.5 and box[1] == 1:
					x_min = int(box[3] * input_image_object_detection.shape[3])
					y_min = int(box[4] * input_image_object_detection.shape[2])
					x_max = int(box[5] * input_image_object_detection.shape[3])
					y_max = int(box[6] * input_image_object_detection.shape[2])


			# A set of three inference results is obtained
			results = {
				name: self.infer_request.get_tensor(name).data[:]
				for name in {"features", "heatmaps", "pafs"}
			}
			# Get the results
			results = (results["features"][0], results["heatmaps"][0], results["pafs"][0])
			poses_3d, _ = parse_poses(results, 1, self.stride, self.focal_length, True)
			if len(poses_3d) > 0:
						# From here, you can rotate the 3D point positions using the function "draw_poses",
						# or you can directly make the correct mapping below to properly display the object image on the screen
						poses_3d_copy = poses_3d.copy()
						x = poses_3d_copy[:, 0::4]
						y = poses_3d_copy[:, 1::4]
						z = poses_3d_copy[:, 2::4]
						poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = (
							-z + np.ones(poses_3d[:, 2::4].shape) * 200,
							-y + np.ones(poses_3d[:, 2::4].shape) * 100,
							-x,
						)

						poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
						poses_3d = poses_3d.astype(float)
						if len(offset) == 0:
							# Find the center of the skeleton
							center = np.mean(poses_3d, axis=(0, 1))

							# Compute the offset to move the skeleton to the origin
							offset = -center

						# Apply the offset to all points of the skeleton
						poses_3d += offset
			poses_3d /= 100

			for pose_3d in poses_3d:
				frame_pose = {
					"frame": i,
					"pose": {
						"neck": {"y":pose_3d[0][0], "z":pose_3d[0][1], "x":pose_3d[0][2]},
						"r_shoulder": {"y":pose_3d[9][0], "z":pose_3d[9][1], "x":pose_3d[9][2]},
						"r_elbow": {"y":pose_3d[10][0], "z":pose_3d[10][1], "x":pose_3d[10][2]},
						"r_wrist": {"y":pose_3d[11][0], "z":pose_3d[11][1], "x":pose_3d[11][2]},
						"l_shoulder": {"y":pose_3d[3][0], "z":pose_3d[3][1], "x":pose_3d[3][2]},
						"l_elbow": {"y":pose_3d[4][0], "z":pose_3d[4][1], "x":pose_3d[4][2]},
						"l_wrist": {"y":pose_3d[5][0], "z":pose_3d[5][1], "x":pose_3d[5][2]},
						"l_eye": {"y":pose_3d[15][0], "z":pose_3d[15][1], "x":pose_3d[15][2]},
						"l_ear": {"y":pose_3d[16][0], "z":pose_3d[16][1], "x":pose_3d[16][2]},
						"r_eye": {"y":pose_3d[17][0], "z":pose_3d[17][1], "x":pose_3d[17][2]},
						"r_ear": {"y":pose_3d[18][0], "z":pose_3d[18][1], "x":pose_3d[18][2]},
						"nose": {"y":pose_3d[1][0], "z":pose_3d[1][1], "x":pose_3d[1][2]},
						"l_hip": {"y":pose_3d[6][0], "z":pose_3d[6][1], "x":pose_3d[6][2]},
						"l_knee": {"y":pose_3d[7][0], "z":pose_3d[7][1], "x":pose_3d[7][2]},
						"l_ankle": {"y":pose_3d[8][0], "z":pose_3d[8][1], "x":pose_3d[8][2]},
						"r_hip": {"y":pose_3d[12][0], "z":pose_3d[12][1], "x":pose_3d[12][2]},
						"r_knee": {"y":pose_3d[13][0], "z":pose_3d[13][1], "x":pose_3d[13][2]},
						"r_ankle": {"y":pose_3d[14][0], "z":pose_3d[14][1], "x":pose_3d[14][2]}
					},
					"box": {
						"x_min": x_min,
						"y_min": y_min,
						"x_max": x_max,
						"y_max": y_max
					}
				}
				poses.append(frame_pose)
			i += 1
			if cv2.waitKey(1) == ord('q'):
				break
		cap.release()
		cv2.destroyAllWindows()
		poses = self.smoothing(poses, smoothing_window)
		with open(output_path, "w") as f:
			json.dump(poses, f)

def smoothing(self, data: list, window_size: int):
	"""
+    Applies smoothing to the input data for each bone name using a moving average filter.
+
+    :param data: A list of dictionaries representing the data to be smoothed.
+    :type data: list
+    :param window_size: The size of the window to use for the moving average filter.
+    :type window_size: int
+    :return: A new list of dictionaries with smoothed data for each bone name.
+    :rtype: list
+    """
	new_data = data.copy()
	for name in self.bone_name:
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


def main():
	"""
	Main function for running pose estimation on a given video file using the Open Model Zoo.

	Args:
		None.

	Returns:
		None.

	Raises:
		None.
	"""
	base_model_dir = "model"
	# model name as named in Open Model Zoo
	model_name = "human-pose-estimation-3d-0001"
	# selected precision (FP32, FP16)
	precision = "FP32"
	pose_estimator = PoseEstimator(base_model_dir, model_name, precision)
	pose_estimator.estimate('0327.mp4', "poses.json")

if __name__ == "__main__":
	main()