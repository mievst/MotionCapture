"""Скрипт устанавливающий координаты рига"""

import bpy
import json
import mathutils
import math


class RigHandler:
    """Rig Handler class."""

    def __init__(self, rig_name : str,):
        """Initializes a RigHandler"""
        # выбираем объект, который нужно переместить
        self.rig = bpy.data.objects[rig_name]
        self.rel = 1
        self.base_bone = self.rig.pose.bones["torso"]
        self.base_bone_loc = None

    base_rig_bones_openvino = {
        #"neck": "neck",
        #"right_shoulder": "shoulder.R",
        #"r_elbow": "forearm_tweak.R",
        "r_wrist": "hand_ik.R",
        #"left_shoulder": "shoulder.L",
        #"l_elbow": "forearm_tweak.L",
        "l_wrist": "hand_ik.L",
        "l_eye": None,
        "l_ear": None,
        #"r_eye": None,
        #"r_ear": None,
        #"nose": None,
        "l_hip": None,
        "l_knee": "shin_tweak.L",
        "l_ankle": "foot_ik.L",
        "r_hip": None,
        "r_knee": "shin_tweak.R",
        "r_ankle": "foot_ik.R",
    }

    base_rig_bone_mediapipe = {
    "nose": None,
    "left_eye_inner": None,
    "left_eye": None,
    "left_eye_outer": None,
    "right_eye_inner": None,
    "right_eye": None,
    "right_eye_outer": None,
    "left_ear": None,
    "right_ear": None,
    "mouth_left": None,
    "mouth_right": None,
    #"left_shoulder": "shoulder.R",
    #"right_shoulder": "shoulder.L",
    #"left_elbow": "forearm_tweak.R",
    #"right_elbow": "forearm_tweak.L",
    "left_wrist": "hand_ik.R",
    "right_wrist": "hand_ik.L",
    "left_pinky": None,
    "right_pinky": None,
    "left_index": None,
    "right_index": None,
    "left_thumb": None,
    "right_thumb": None,
    "left_hip": None,
    "right_hip": None,
    #"left_knee": "shin_tweak.R",
    #"right_knee": "shin_tweak.L",
    "left_ankle": "foot_ik.R",
    "right_ankle": "foot_ik.L",
    "left_heel": None,
    "right_heel": None,
    "left_foot_index": None,
    "right_foot_index": None
    }

    def set_bone_loc(self, bone_name, location):
        """Set the bone location to the given location.

        Args:
            bone (bpy.types.Bone): The bone to set the location.
            location (mathutils.Vector): The location to set.
        """
        bone = self.rig.pose.bones[bone_name]
        matrix_world = self.rig.matrix_world
        # Получаем матрицу преобразования конечной точкой относительно центра компонент"""
        location += self.rig.location
        # Получаем матрицу преобразования кости относительно центра координат
        matrix = matrix_world @ bone.matrix
        _, bone_rot, bone_sca = matrix.decompose()
        mat_out = mathutils.Matrix.LocRotScale(location, bone_rot, bone_sca)
        bone.matrix = matrix_world.inverted() @ mat_out
        # Обновляем костную систему
        bpy.context.view_layer.update()
        bone.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)


    def set_bone_rot_from_parent(self, bone, parent_bone):
        # Получаем матрицу преобразования кости относительно центра объекта
        matrix = self.rig.matrix_world @ bone.matrix
        parent_matrix = self.rig.matrix_world @ parent_bone.matrix
        bone_loc, _, bone_sca = matrix.decompose()
        _, parent_bone_rot, _ = parent_matrix.decompose()
        mat_out = mathutils.Matrix.LocRotScale(bone_loc, parent_bone_rot, bone_sca)
        bone.matrix = self.rig.matrix_world.inverted() @ mat_out
        # Обновляем костную систему
        bpy.context.view_layer.update()
        bone.keyframe_insert(data_path="rotation_quaternion", frame=bpy.context.scene.frame_current)

    def set_bone_rot(self, bone, quaternion):
        # Получаем матрицу преобразования кости относительно центра объекта
        matrix = self.rig.matrix_world @ bone.matrix
        bone_loc, _, bone_sca = matrix.decompose()
        mat_out = mathutils.Matrix.LocRotScale(bone_loc, quaternion, bone_sca)
        bone.matrix = self.rig.matrix_world.inverted() @ mat_out
        # Обновляем костную систему
        bpy.context.view_layer.update()
        bone.keyframe_insert(data_path="rotation_quaternion", frame=bpy.context.scene.frame_current)

    def set_torso_data(self, pose : dict, l_hip_bone_name : str, r_hip_bone_name : str, neck_bone_name : str):
        l_hip = self.get_loc(pose, l_hip_bone_name)
        r_hip = self.get_loc(pose, r_hip_bone_name)
        neck = self.get_loc(pose, neck_bone_name)
        torso_bone = self.rig.pose.bones["torso"]

        # Найдите центр между первой и второй точками.
        center = (l_hip + r_hip) / 2

        x_vec = mathutils.Vector((-1, 0, 0))
        y_vec = mathutils.Vector((0, 1, 0))
        z_vec = mathutils.Vector((0, 0, 1))

        bone_vec = center - r_hip
        neck_vec = neck - center

        x_angle = x_vec.angle(bone_vec)
        y_angle = y_vec.cross(bone_vec).angle(neck_vec) #y_vec.angle(neck_vec) # Угол поворота вокруг оси Y всегда равен 0 для кости torso
        z_angle = z_vec.angle(neck_vec)

        # Вычисляем углы поворота кости torso
        torso_bone.rotation_mode = 'XYZ'
        torso_bone.rotation_euler = mathutils.Euler((x_angle, y_angle, z_angle), 'XYZ')
        bpy.context.view_layer.update()

        self.set_bone_loc("torso", center)
        torso_bone.keyframe_insert(data_path="rotation_euler", frame=bpy.context.scene.frame_current)
        #self.set_bone_rot(self.rig.pose.bones["torso"], new_quaternion)

    def set_shoulder_data(self, pose : dict, l_shoulder_bone_name : str, r_shoulder_bone_name : str):
        l_shoulder = self.get_loc(pose, l_shoulder_bone_name)
        r_shoulder = self.get_loc(pose, r_shoulder_bone_name)

        center = (l_shoulder + r_shoulder) / 2
        bone = self.rig.pose.bones["chest"]
        center += self.rig.location
        # Получаем матрицу преобразования кости относительно центра координат
        matrix = self.rig.matrix_world @ self.rig.pose.bones["torso"].matrix.inverted() @ bone.matrix
        _, bone_rot, bone_sca = matrix.decompose()
        mat_out = mathutils.Matrix.LocRotScale(center, bone_rot, bone_sca)
        bone.matrix = self.rig.matrix_world.inverted()@ self.rig.pose.bones["torso"].matrix @ mat_out
        # Обновляем костную систему
        bpy.context.view_layer.update()
        bone.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)

    def get_loc(self, pose : dict, bone_name : str) :
        """
        returns the location of the bone

        Args:
            pose (dict): dictionary with pose data
            bone_name (str): bone name
            base_bone_name (str): base bone name

        Returns:
            mathutils.Vector: new bone location
        """
        bone_vector = mathutils.Vector((
            pose["pose"][bone_name]["x"],
            pose["pose"][bone_name]["y"],
            pose["pose"][bone_name]["z"]))

        new_location = (bone_vector - self.rig.location) * self.rel
        return new_location

    def set_frame(self, pose : dict):
        """
        Sets the pose of the rig to the given pose.

        Args:
            pose (dict): A dictionary containing the pose information.
                frame (int): The frame number.
                pose (dict): A dictionary containing the pose information.

        Returns:
            None.
        """

        bpy.context.scene.frame_set(pose["frame"])
        self.set_torso_data(pose, "l_hip", "r_hip", "neck")
        #self.set_shoulder_data(pose, "l_shoulder", "r_shoulder")
        for key, val in self.base_rig_bones_openvino.items():
            if val == None:
                continue
            new_location = self.get_loc(pose, key)
            self.set_bone_loc(val, new_location)
        self.set_bone_rot_from_parent(self.rig.pose.bones["hand_ik.R"], self.rig.pose.bones["MCH-forearm_ik.R"])
        self.set_bone_rot_from_parent(self.rig.pose.bones["hand_ik.L"], self.rig.pose.bones["MCH-forearm_ik.L"])

    def find_relation(self, poses : list[dict]):
        high = (poses[0]["box"]["y_max"] - poses[0]["box"]["y_min"]) * 0.6
        self.rel = self.rig.dimensions.z / high
        pass

    def set_frame_list(self, poses : list[dict]):
        """
        Sets the location keyframe for each pose in given list of poses.

        Args:
            poses (list[dict]): A list of dictionary containing pose information.

        Returns:
            None
        """
        self.find_relation(poses)
        for pose in poses:
            self.set_frame(pose)
        self.rig.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)

def main():
    poses = []
    with open("C:\\Users\\mievst\\Desktop\\masters\\pose_export\\poses.json", "r") as f:
        poses = json.loads(f.read())
    poses.sort(key = lambda frame: frame["frame"])
    handler = RigHandler("rig")
    handler.set_frame_list(poses)


def fake_main():
    rig = bpy.data.objects["rig"]
    bone = rig.pose.bones["neck"]
    new_location = Vector((0.006395, 0.377164, 1.02685))
    handler = RigHandler("rig")
    handler.set_bone_loc(bone=bone, new_location=new_location)

main()
#fake_main()
