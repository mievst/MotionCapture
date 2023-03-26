import bpy

# задаем желаемое время анимации (в секундах)
time_seconds = 2.5

# устанавливаем текущее время анимации
bpy.context.scene.frame_set(int(time_seconds * bpy.context.scene.render.fps))

# устанавливаем контрольный кадр для всех выбранных объектов
selected_objs = bpy.context.selected_objects
for obj in selected_objs:
    obj.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)