import os

def delete_txt_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                os.remove(file_path)

# 示例文件夹路径
folder_path = 'output'

# 删除文件夹中的所有 .txt .jpg文件
delete_txt_files(folder_path)