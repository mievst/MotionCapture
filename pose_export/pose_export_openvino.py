from pathlib import Path
from openvino.runtime import Core
import cv2
import numpy as np
import sys
from pathlib import Path
import subprocess

sys.path.append("./engine")
#import engine.engine3js as engine
from engine.parse_poses import parse_poses

class PoseEstimator:

	def __init__(self, model_dir, model_name, precision):
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