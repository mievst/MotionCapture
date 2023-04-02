"""скрипт устанавливающий координаты рига"""

import bpy
import json
import mathutils

def set_bone_2(rig, bone, location):
    # Получаем матрицу преобразования кости относительно центра объекта
    matrix = rig.matrix_world @ bone.matrix

    bone_loc, bone_rot, bone_sca = matrix.decompose()

    mat_out = mathutils.Matrix.LocRotScale(location, bone_rot, bone_sca)

    bone.matrix = rig.matrix_world.inverted() @ mat_out

    # Обновляем костную систему
    bpy.context.view_layer.update()
    bone.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)

def set_bone(rig, bone, location):
    # Получаем матрицу преобразования кости относительно центра объекта
    matrix = rig.matrix_world @ bone.matrix

    re_matrix = rig.matrix_world.inverted() @ matrix

    bone_loc, bone_rot, bone_sca = bone.matrix.decompose()
    rig_loc, rig_rot, rig_sca = rig.matrix_world.decompose()
    bone_w_loc, bone_w_rot, bone_w_sca = matrix.decompose()

    # Считаем новый вектор, относительно которого нужно задать координаты
    origin = rig.matrix_world @ bone.head

    # Считаем поворот кости в глобальной системе координат
    rotation = rig.matrix_world.to_quaternion() @ bone.matrix.to_quaternion()

    # Складываем новые координаты с координатами центра кости
    new_matrix_translation = location - origin

    # Изменяем матрицу преобразования кости, чтобы она соответствовала новым координатам
    matrix = Matrix.Translation(new_matrix_translation) @ rotation.to_matrix().to_4x4()

    # Применяем изменения к кости
    bone.matrix_basis = rig.matrix_world.inverted() @ matrix

    # Обновляем костную систему
    bpy.context.view_layer.update()
    bone.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)

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
        #set_bone_absolute_location(rig, "torso", hips_loc)

        for key, val in base_rig_bones.items():
            if val == None:
                continue
            bone = rig.pose.bones[val]

            new_location = mathutils.Vector((pose["pose"][key]["x"] / 100, pose["pose"][key]["y"] / 100, pose["pose"][key]["z"] / 100))

            set_bone_2(rig, bone, new_location)
            #set_bone_location(bone, new_location)

        rig.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)


def fake_main():
    rig = bpy.data.objects["rig"]
    bone = rig.pose.bones["neck"]
    new_location = Vector((0.006395, 0.377164, 1.02685))
    set_bone(rig, bone, new_location)

main()
#fake_main()
