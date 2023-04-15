"""Скрипт устанавливающий координаты рига"""

import bpy
import json
import mathutils

def set_bone_loc(rig, bone, location):
    """Set the bone location to the given location.

    Args:
        rig (bpy.types.Object): The rig object.
        bone (bpy.types.Bone): The bone to set the location.
        location (mathutils.Vector): The location to set.
    """

    # Получаем матрицу преобразования конечной точкой относительно центра компонент"""
    base_bone = rig.pose.bones["neck"]

    base_loc, _, _ = (rig.matrix_world @ base_bone.matrix).decompose()

    location += base_loc

    # Получаем матрицу преобразования кости относительно центра координат
    matrix = rig.matrix_world @ bone.matrix

    bone_loc, bone_rot, bone_sca = matrix.decompose()

    mat_out = mathutils.Matrix.LocRotScale(location, bone_rot, bone_sca)

    bone.matrix = rig.matrix_world.inverted() @ mat_out

    # Обновляем костную систему
    bpy.context.view_layer.update()
    bone.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)


def set_bone_rot(rig, bone, parent_bone):
    """Set the bone rotation to the given rotation.

    Args:
        rig (bpy.types.Object): The rig object.
        bone (bpy.types.Bone): The bone to set the rotation.
        parent_bone (bpy.types.Bone): The parent bone of the bone to set the rotation.
    """
    # Получаем матрицу преобразования кости относительно центра объекта
    matrix = rig.matrix_world @ bone.matrix
    parent_matrix = rig.matrix_world @ parent_bone.matrix

    bone_loc, bone_rot, bone_sca = matrix.decompose()
    paremt_bone_loc, parent_bone_rot, parent_bone_sca = parent_matrix.decompose()

    mat_out = mathutils.Matrix.LocRotScale(bone_loc, parent_bone_rot, bone_sca)

    bone.matrix = rig.matrix_world.inverted() @ mat_out

    # Обновляем костную систему
    bpy.context.view_layer.update()
    bone.keyframe_insert(data_path="rotation_quaternion", frame=bpy.context.scene.frame_current)

def get_hips_loc(l_hip, r_hip):
    l_hip_x, l_hip_y, l_hip_z = (l_hip["x"] / 100, l_hip["y"] / 100, l_hip["z"] / 100)
    r_hip_x, r_hip_y, r_hip_z = (r_hip["x"] / 100, r_hip["y"] / 100, r_hip["z"] / 100)

    middle_x = (l_hip_x + r_hip_x) / 2
    middle_y = (l_hip_y + r_hip_y) / 2
    middle_z = (l_hip_z + r_hip_z) / 2
    return mathutils.Vector((middle_x, middle_y, middle_z))

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

def main():
    poses = []
    with open("C:\\Users\\mievst\\Desktop\\masters\\pose_export\\poses.json", "r") as f:
        poses = json.loads(f.read())

    poses.sort(key = lambda frame: frame["frame"])

    # выбираем объект, который нужно переместить
    rig = bpy.data.objects["rig"]

    for i in range(0, len(poses), 5):
        pose = poses[i]
    #for pose in poses:

        bpy.context.scene.frame_set(pose["frame"])

        #hips_loc = get_hips_loc(pose["pose"]["l_hip"], pose["pose"]["r_hip"])

        #set_bone(rig, rig.pose.bones["torso"], hips_loc)

        for key, val in base_rig_bones.items():
            if val == None:
                continue
            bone = rig.pose.bones[val]

            x = (pose["pose"][key]["x"] - pose["pose"]["neck"]["x"]) / 100
            y = (pose["pose"][key]["y"] - pose["pose"]["neck"]["y"]) / 100
            z = (pose["pose"][key]["z"] - pose["pose"]["neck"]["z"]) / 100

            new_location = mathutils.Vector((x, y, z))

            set_bone_loc(rig, bone, new_location)

        set_bone_rot(rig, rig.pose.bones["hand_ik.R"], rig.pose.bones["MCH-forearm_ik.R"])
        set_bone_rot(rig, rig.pose.bones["hand_ik.L"], rig.pose.bones["MCH-forearm_ik.L"])

    rig.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)


def fake_main():
    rig = bpy.data.objects["rig"]
    bone = rig.pose.bones["neck"]
    new_location = Vector((0.006395, 0.377164, 1.02685))
    set_bone_loc(rig, bone, new_location)

main()
#fake_main()
