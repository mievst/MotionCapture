"""Скрипт устанавливающий координаты рига"""

import bpy
import json
import mathutils

class RigHandler:
    """Rig Handler class."""

    def __init__(self, rig_name : str,):
        """Initializes a RigHandler"""
        # выбираем объект, который нужно переместить
        self.rig = bpy.data.objects[rig_name]

    base_rig_bones = {
        #"neck": "neck",
        #"r_shoulder": "shoulder.R",
        #"r_elbow": "forearm_tweak.R",
        "r_wrist": "hand_ik.R",
        #"l_shoulder": "shoulder.L",
        #"l_elbow": "forearm_tweak.L",
        "l_wrist": "hand_ik.L",
        "l_eye": None,
        "l_ear": None,
        #"r_eye": None,
        #"r_ear": None,
        #"nose": None,
        #"l_hip": None,
        "l_knee": "shin_tweak.L",
        "l_ankle": "foot_ik.L",
        #"r_hip": None,
        "r_knee": "shin_tweak.R",
        "r_ankle": "foot_ik.R",
    }

    def set_bone_loc(self, bone, location):
        """Set the bone location to the given location.

        Args:
            bone (bpy.types.Bone): The bone to set the location.
            location (mathutils.Vector): The location to set.
        """

        # Получаем матрицу преобразования конечной точкой относительно центра компонент"""
        base_bone = self.rig.pose.bones["neck"]

        base_loc, _, _ = (self.rig.matrix_world @ base_bone.matrix).decompose()

        location += base_loc

        # Получаем матрицу преобразования кости относительно центра координат
        matrix = self.rig.matrix_world @ bone.matrix

        _, bone_rot, bone_sca = matrix.decompose()

        mat_out = mathutils.Matrix.LocRotScale(location, bone_rot, bone_sca)

        bone.matrix = self.rig.matrix_world.inverted() @ mat_out

        # Обновляем костную систему
        bpy.context.view_layer.update()
        bone.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)


    def set_bone_rot(self, bone, parent_bone):
        """Set the bone rotation to the given rotation.

        Args:
            bone (bpy.types.Bone): The bone to set the rotation.
            parent_bone (bpy.types.Bone): The parent bone of the bone to set the rotation.
        """
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

    def get_hips_loc(l_hip, r_hip):
        """Get the hips location.

        Args:
            l_hip (dict): The left hip.
            r_hip (dict): The right hip.

        Returns:
            mathutils.Vector: The hips location.
        """

        l_hip_x, l_hip_y, l_hip_z = (l_hip["x"] / 100, l_hip["y"] / 100, l_hip["z"] / 100)
        r_hip_x, r_hip_y, r_hip_z = (r_hip["x"] / 100, r_hip["y"] / 100, r_hip["z"] / 100)

        middle_x = (l_hip_x + r_hip_x) / 2
        middle_y = (l_hip_y + r_hip_y) / 2
        middle_z = (l_hip_z + r_hip_z) / 2
        return mathutils.Vector((middle_x, middle_y, middle_z))

    def set_frame(self, pose : dict):
        bpy.context.scene.frame_set(pose["frame"])

        #hips_loc = get_hips_loc(pose["pose"]["l_hip"], pose["pose"]["r_hip"])
        for key, val in self.base_rig_bones.items():
            if val == None:
                continue
            bone = self.rig.pose.bones[val]

            x = (pose["pose"][key]["x"] - pose["pose"]["neck"]["x"]) / 100
            y = (pose["pose"][key]["y"] - pose["pose"]["neck"]["y"]) / 100
            z = (pose["pose"][key]["z"] - pose["pose"]["neck"]["z"]) / 100

            new_location = mathutils.Vector((x, y, z))

            self.set_bone_loc(bone, new_location)

        self.set_bone_rot(self.rig.pose.bones["hand_ik.R"], self.rig.pose.bones["MCH-forearm_ik.R"])
        self.set_bone_rot(self.rig.pose.bones["hand_ik.L"], self.rig.pose.bones["MCH-forearm_ik.L"])

    def set_frame_list(self, poses : list[dict]):
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
