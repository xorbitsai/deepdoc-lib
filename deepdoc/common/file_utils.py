import os

# 去除对 ragflow 环境变量的依赖，改为基于 deepdoc 项目根目录的相对路径计算
def get_project_base_directory(*args):
    # 计算 deepdoc 项目根目录，假设 depend 文件夹在 deepdoc 根目录下
    deepdoc_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    if args:
        return os.path.join(deepdoc_root, *args)
    return deepdoc_root



