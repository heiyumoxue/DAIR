# from PIL import Image
# from os import listdir
# import numpy as np
#
#
# def resize_img(input_path, out_path, x, y):
#     fp = open(input_path, 'rb')
#     pic = Image.open(fp)
#     pic_array = np.array(pic)
#     fp.close()
#     img = Image.fromarray(pic_array)
#     print("修改前: ", img.size)
#     new_img = img.resize((x, y))
#     new_img.save(out_path)
#     print("修改后: ", new_img.size)
#
#
# if __name__ == '__main__':
#     inpath = r'F:\2023\Images\datasets\train3'  # 在此输入图片输入路径
#     outpath = r'F:\2023\Images\datasets\train3'  # 在此输入图片输出路径
#     x = 255  # 图片水平长度
#     y = 255  # 图片垂直长度
#     # x = 1023  # 图片水平长度
#     # y = 1023  # 图片垂直长度
#
#     for i in listdir(inpath):
#         resize_img(inpath + '\\' + i, outpath + '\\' + i, x, y)
#         print("--------------------")






# from PIL import Image
# import os
#
# # 设置要搜索的文件夹路径
# folder_path = "F:/2023/Images/agricultural"
#
# # 获取文件夹中所有文件的列表
# files = os.listdir(folder_path)
#
# # 遍历所有文件
# for file_name in files:
#     # 获取文件路径
#     file_path = os.path.join(folder_path, file_name)
#     # 判断文件是否为图像文件
#     if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".tif"):
#         # 打开图像文件
#         img = Image.open(file_path)
#         # 获取图像的宽度和高度
#         width, height = img.size
#         # 输出图像的分辨率
#         print(f"{file_name}: {width} x {height}")


# #####判断指定文件夹中所有图片文件的分辨率是否符合指定的宽度和高度要求。如果图像文件的分辨率不符合要求，将输出其文件名和实际分辨率。
# from PIL import Image
# import os
#
# # 设置要搜索的文件夹路径
# folder_path  = "F:/2023/Images"
#
# # 设置要求的宽度和高度
# width = 256
# height = 256
#
# # 获取文件夹中所有文件的列表
# files = os.listdir(folder_path)
#
# # 遍历所有文件
# for file_name in files:
#     # 获取文件路径
#     file_path = os.path.join(folder_path, file_name)
#     # 判断文件是否为图像文件
#     if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".tif"):
#         # 打开图像文件
#         img = Image.open(file_path)
#         # 获取图像的宽度和高度
#         img_width, img_height = img.size
#         # 判断图像是否符合要求
#         if img_width != width or img_height != height:
#             # 输出不符合要求的图像文件名和实际分辨率
#             print(f"{file_name}: {img_width} x {img_height}")





# ##修改指定文件夹中所有图片文件的分辨率为指定的宽度和高度。请将脚本中的“folder_path”替换为要修改的文件夹路径，“new_width”和“new_height”替换为指定的宽度和高度。
from PIL import Image
import os

# 设置要修改的文件夹路径
folder_path  = "F:/2023/Images/datasets/train3/"

# 设置新的宽度和高度
new_width = 255
new_height = 255

# 获取文件夹中所有文件的列表
files = os.listdir(folder_path)

# 遍历所有文件
for file_name in files:
    # 获取文件路径
    file_path = os.path.join(folder_path, file_name)
    # 判断文件是否为图像文件
    if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".tif"):
        # 打开图像文件
        img = Image.open(file_path)
        # 获取原始图像的宽度和高度
        width, height = img.size
        # 计算新的图像高度
        new_img_height = int(new_width * height / width)
        # 调整图像大小
        img = img.resize((new_width, new_img_height))
        # 保存修改后的图像
        img.save(file_path)


# #用于检索整个文件夹中所有图像的分辨率，并找出分辨率不同的图像并将其存储到另一个文件夹：
# import os
# from PIL import Image
#
# # 设置要查找的文件夹路径和目标分辨率
# folder_path  = "F:/2023/Images"
# target_resolution = (256, 256)
#
# # 设置要保存不同分辨率图像的文件夹路径
# diff_res_folder = "F:/2023/test1"
#
# # 创建目标文件夹（如果不存在）
# if not os.path.exists(diff_res_folder):
#     os.makedirs(diff_res_folder)
#
# # 遍历文件夹中的所有图像文件
# for filename in os.listdir(folder_path):
#     filepath = os.path.join(folder_path, filename)
#
#     # 仅处理图像文件
#     if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
#
#         # 打开图像文件并获取其分辨率
#         with Image.open(filepath) as im:
#             resolution = im.size
#
#             # 如果分辨率与目标分辨率不同，则将其保存到另一个文件夹
#             if resolution != target_resolution:
#                 diff_res_filepath = os.path.join(diff_res_folder, filename)
#                 im.save(diff_res_filepath)
#                 print(f"{filename}: {resolution}")






# import os
# import shutil
#
# # 设置待整理的文件夹路径和目标文件夹路径
# src_dir = "F:/2023/Images/HR"
# dst_dir = "F:/2023/Images/GT"
# # 遍历待整理的文件夹中所有的文件
# for filename in os.listdir(src_dir):
#     # 获取文件名前四个单词
#     words = filename.split()[:4]
#     prefix = "_".join(words)
#
#     # 如果该文件前4个单词相同，并且前缀名称不为空
#     if len(words) == 4 and prefix:
#         # 构造目标文件夹路径
#         prefix_dir = os.path.join(dst_dir, prefix)
#         if not os.path.exists(prefix_dir):
#             os.makedirs(prefix_dir)
#
#         # 将文件移动到目标文件夹中
#         src_file = os.path.join(src_dir, filename)
#         dst_file = os.path.join(prefix_dir, filename)
#         shutil.move(src_file, dst_file)


###实现从一个文件夹中找出前五个字母一样的图片，并将这些图片放入以前五个字母命名的文件夹中：
# import os
# import shutil
#
# # 指定待处理的文件夹
# folder_path = src_dir = "F:/2023/Images/data/test/X4"
#
# # 遍历该文件夹中的所有文件
# for file_name in os.listdir(folder_path):
#     # 仅处理JPEG和PNG格式的图片
#     if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.tif'):
#         # 获取文件名的前五个字符
#         prefix = file_name[:5]
#         # 构造目标文件夹路径
#         target_folder = os.path.join(folder_path, prefix)
#         # 如果目标文件夹不存在，则创建该文件夹
#         if not os.path.exists(target_folder):
#             os.mkdir(target_folder)
#         # 将图片移动到目标文件夹中
#         shutil.move(os.path.join(folder_path, file_name), os.path.join(target_folder, file_name))


# import os
#
# # 指定待处理的文件夹
# folder = "F:/2023/Images/HR1"
#
# # 定义每隔几张删除一张图像
# interval = 10
#
# # 计数器
# counter = 0
#
# # 遍历该文件夹中的所有文件
# for file_name in os.listdir(folder):
#     # 仅处理JPEG和PNG格式的图片
#     if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.png'):
#         # 判断当前文件是否是每隔 interval 张图像
#         if counter % interval == 0:
#             # 删除符合条件的图片
#             os.remove(os.path.join(folder, file_name))
#         # 增加计数器的值
#         counter += 1


# import os
# from PIL import Image
#
# # 源文件夹路径
# src_folder_path =r'G:\自己的论文数据\对比结果图\补充X4的遥感图像\airplane37'  # 在此输入图片输入路径
#
# # 目标文件夹路径
# dst_folder_path = r'G:\自己的论文数据\对比结果图\补充X4的遥感图像\airplane37'  # 在此输入图片输出路径
#
# # 遍历源文件夹中所有的文件，如果文件包含图片，则将其转换为PNG格式并保存到目标文件夹中
# for filename in os.listdir(src_folder_path):
#     # 检查文件是否为图片格式
#     if filename.endswith(".jpg") or filename.endswith(".tif") or filename.endswith(".png") or filename.endswith(".bmp") or filename.endswith(".gif"):
#         # 打开图片
#         img = Image.open(os.path.join(src_folder_path, filename))
#
#         # 生成目标文件名
#         dst_filename = os.path.splitext(filename)[0] + ".png"
#
#         # 保存图片到目标文件夹
#         img.save(os.path.join(dst_folder_path, dst_filename))
#
# print("转换完成！")

