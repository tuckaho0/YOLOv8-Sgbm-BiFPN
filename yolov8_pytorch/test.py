# # test_paths.py
# import os
# from pathlib import Path
#
#
# def check_paths():
#     current_file = Path(__file__).resolve()
#     root_dir = current_file.parent.parent
#     classes_path = root_dir /"yolov8_pytorch"/ "model_data" / "voc_classes.txt"
#
#     print("项目根目录:", root_dir)
#     print("类别文件路径:", classes_path)
#     print("文件存在:", os.path.exists(classes_path))
#
#
# if __name__ == "__main__":
#     check_paths()
with open(r'D:\YOLO\YOLOv8-Sgbm - 1\yolov8_pytorch\model_data\voc_classes.txt', 'r') as f:
    classes = f.readlines()
print(f"检测到{len(classes)}个类别：{classes}")