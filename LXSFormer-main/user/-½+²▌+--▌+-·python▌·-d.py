import os
import argparse
import cv2
# 双三次下采样的代码-------------------这个是可以直接用来下采样的代码
# 数据集的HR和LR名字要一一对应，否则找不到对应的路径，用已有的程序下采样得到的图片会有后缀，记得改掉
# 数据集的HR和LR名字要一一对应，否则找不到对应的路径，用已有的程序下采样得到的图片会有后缀，记得改掉
# 遥感图像下采样代码   注意文件夹不能有特殊字符
# parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')  #“使用双三次插值将图像缩小2倍”
parser.add_argument("-k", "--keepdims", help="keep original image dimensions in downsampled images",
                    action="store_true")   #在降采样图像中保持原始图像尺寸
parser.add_argument('--hr_img_dir', type=str, default=r'/home/oem/data/dataset/DIV2K/DIV2K_train_HR_sub',
                    help='path to high resolution image dir')  #高分辨率图像目录的路径  这里的路径是本地电路的绝对路径  不能出现中文路径
parser.add_argument('--lr_img_dir', type=str, default=r'/home/oem/data/dataset/DIV2K/DIV2K_train_BIC_sub',
                    help='path to desired output dir for downsampled images')  #下采样图像的所需输出方向的路径
args = parser.parse_args()

hr_image_dir = args.hr_img_dir
lr_image_dir = args.lr_img_dir

print(args.hr_img_dir)
print(args.lr_img_dir)

# create LR image dirs
os.makedirs(lr_image_dir + "/X2", exist_ok=True)
os.makedirs(lr_image_dir + "/X3", exist_ok=True)
os.makedirs(lr_image_dir + "/X4", exist_ok=True)
# os.makedirs(lr_image_dir + "/X6", exist_ok=True)

supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                         ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                         ".tiff")

# Downsample HR images
for filename in os.listdir(hr_image_dir):
    if not filename.endswith(supported_img_formats):
        continue

    name, ext = os.path.splitext(filename)

    # Read HR image
    hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
    hr_img_dims = (hr_img.shape[1], hr_img.shape[0])

    # Blur with Gaussian kernel of width sigma = 1
    hr_img = cv2.GaussianBlur(hr_img, (0, 0), 1, 1)
    # cv2.GaussianBlur(hr_img, (0,0), 1, 1)   其中模糊核这里用的0。两个1分别表示x、y方向的标准差。 可以具体查看该函数的官方文档。
    # Downsample image 2x
    lr_image_2x = cv2.resize(hr_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_image_2x = cv2.resize(lr_image_2x, hr_img_dims, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(os.path.join(lr_image_dir + "/X2", filename.split('.')[0] + 'x2' + ext), lr_image_2x)

    # Downsample image 3x
    lr_img_3x = cv2.resize(hr_img, (0, 0), fx=(1 / 3), fy=(1 / 3),
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_3x = cv2.resize(lr_img_3x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X3", filename.split('.')[0] + 'x3' + ext), lr_img_3x)

    # Downsample image 4x
    lr_img_4x = cv2.resize(hr_img, (0, 0), fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_4x = cv2.resize(lr_img_4x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X4", filename.split('.')[0] + 'x4' + ext), lr_img_4x)

    # Downsample image 6x
    # lr_img_6x = cv2.resize(hr_img, (0, 0), fx=1 / 6, fy=1 / 6,
    #                        interpolation=cv2.INTER_CUBIC)
    # if args.keepdims:
    #     lr_img_4x = cv2.resize(lr_img_6x, hr_img_dims,
    #                            interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(lr_image_dir + "/X6", filename.split('.')[0] + 'x6' + ext), lr_img_6x)


