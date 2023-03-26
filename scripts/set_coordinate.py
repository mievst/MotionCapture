"""скрипт устанавливающий координаты рига"""
import bpy
import json
from mathutils import Vector

def set_bone(rig, bone, location):
    # Получаем матрицу преобразования кости относительно центра объекта
    matrix = rig.matrix_world @ bone.matrix

    # Считаем новый вектор, относительно которого нужно задать координаты
    origin = rig.matrix_world @ bone.head

    # Складываем новые координаты с координатами центра кости
    new_matrix_translation = location - origin

    # Изменяем матрицу преобразования кости, чтобы она соответствовала новым координатам
    matrix.translation = new_matrix_translation

    # Применяем изменения к кости
    bone.matrix_basis = rig.matrix_world.inverted() @ matrix

    # Обновляем костную систему
    bpy.context.view_layer.update()
    bone.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)


def get_hips_loc(l_hip, r_hip):
    l_hip_x, l_hip_y, l_hip_z = (l_hip["x"], l_hip["y"], l_hip["z"])
    r_hip_x, r_hip_y, r_hip_z = (r_hip["x"], r_hip["y"], r_hip["z"])

    middle_x = (l_hip_x + r_hip_x) / 2
    middle_y = (l_hip_y + r_hip_y) / 2
    middle_z = (l_hip_z + r_hip_z) / 2
    return Vector((middle_x, middle_y, middle_z))

base_rig_bones = {
    "neck": "neck",
    "r_shoulder": "shoulder.R",
    "r_elbow": "forearm_tweak.R",
    "r_wrist": "hand_ik.R",
    "l_shoulder": "shoulder.L",
    "l_elbow": "forearm_tweak.L",
    "l_wrist": "hand_ik.L",
    "l_eye": None,
    "l_ear": None,
    "r_eye": None,
    "r_ear": None,
    "nose": None,
    "l_hip": None,
    "l_knee": "shin_tweak.L",
    "l_ankle": "foot_ik.L",
    "r_hip": None,
    "r_knee": "shin_tweak.R",
    "r_ankle": "foot_ik.R",
}

poses = []
with open("C:\Users\mievst\Desktop\masters\pose_export\poses.json", "r") as f:
    poses = json.loads(f.read())

poses.sort(key = lambda frame: frame["frame"])

# выбираем объект, который нужно переместить
rig = bpy.data.objects["rig"]

for pose in poses:

    bpy.context.scene.frame_set(pose["frame"])

    hips_loc = get_hips_loc(pose["pose"]["l_hip"], pose["pose"]["r_hip"])

    set_bone(rig, rig.pose.bones["hips"], hips_loc)

    for key, val in base_rig_bones:
        if val == None:
            continue
        bone = rig.pose.bones[val]

        new_location = Vector((pose["pose"][key]["x"], pose["pose"][key]["y"], pose["pose"][key]["z"]))

        set_bone(rig, bone, new_location)

    rig.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)

