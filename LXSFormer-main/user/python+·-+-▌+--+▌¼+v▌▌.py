### self.path = ‘文件地址’  如果不可以  在文件地址前r，r的意思是声明字符串，不用转义处理
### 改成图片所在的文件夹地址
### if item.endswith(’.png’):
### 如果你的图片以jpg结尾 需要将代码中的 png 改为 jpg
### i = 0  如果你想让图片名称从100开始，就可以设置 i = 100


import os
class BatchRename():

    def __init__(self):
        # self.path = r'F:\New_SIDD\SIDD\val\input'  # 图片的路径 target
        self.path = r'F:\Master_CNN\YOLO-Train-detection-model\Bacteria_detection\demo\images\train'  # 图片的路径


    def rename(self):
        filelist = os.listdir(self.path)
        filelist.sort()
        total_num = len(filelist) #获取文件中有多少图片
        i = 1 #文件命名从哪里开始（即命名从哪里开始）
        for item in filelist:
            if item.endswith('.jpg'):#这里的png为文件夹里面的图片格式   如果你的图片格式是jpg就改成jpg  如果是tif就改成tif
                src = os.path.join(self.path, item)
                #这里是排序之后生成的图片格式 如果需要png格式就改成png就可以
                # dst = os.path.join(os.path.abspath(self.path), str(i) + '_x4.png')
                dst = os.path.join(os.path.abspath(self.path), str(i) + '.jpg')  #修改之后的格式
                
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except Exception as e:
                    print(e)
                    print('rename dir fail\r\n')

        print('total %d to rename & converted %d jpgs' % (total_num, i))
if __name__ == '__main__':
    demo = BatchRename()  #创建对象
    demo.rename()   #调用对象的方法
