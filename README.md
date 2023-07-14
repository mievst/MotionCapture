## Human Motion Transfer
This project is a system of scripts that can transfer human motions from a video to a 3D model in Blender3D. It uses computer vision and deep learning techniques to detect and track human poses and apply them to a rigged 3D character.
# How it works
The system consists of three main steps:

 - Pose estimation: The input video is fed to a pose estimation model that outputs a sequence of 3D keypoints for each frame, representing the locations of body parts (such as head, shoulders, elbows, etc.).
 - Pose normalization: The 3D keypoints are normalized by scaling, translating and rotating them to fit a standard coordinate system. This step ensures that the motions are consistent and independent of the camera perspective and the distance from the subject.
 - Pose transfer: The normalized 3D keypoints are mapped to the corresponding 3D bones of the target character in Blender3D using a predefined mapping scheme. The mapping scheme defines how each 3D keypoint corresponds to a 3D bone in terms of name, orientation and length. The 3D bones are then animated by setting their rotation values according to the 2D keypoints.

# Example
Here is an example of how the system works:

<div id="header" align="center">
  <img src="Реальность vs MediaPipe — сделано в Clipchamp.gif"/>
</div>
