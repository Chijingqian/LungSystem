"""
一些预先说明：

原来的训练测试的目录在 /home/liubo/data/graduate/classification_dataset
现在将之前的数据合并在一起在进行重新采样
合并在一起的数据集保存在 /home/liubo/data/graduate/resampled_classification_dataset/total
total 中                     增强后
zero  19 + 6 = 25            25*3 = 75
one   596 + 30 = 626         626 * 3 = 1878
two   166 + 15 = 181         181*3 = 543
three 24 + 6 = 30            30 *3 = 90
four  74 + 10 = 84           84 *3  = 252


本脚本说明：
脚本分两部
1. 在total中生成augumentation
2. 在augumentation中生成 5折交叉验证 subset0 - subset4

"""
import os
import glob
import random
import cv2
import keras as K
import numpy as np
from tqdm import tqdm 
import shutil

config = {}
config["total_dir"] = "/home/liubo/data/graduate/resampled_classification_dataset/total"
config["augumentation_dir"] = "/home/liubo/data/graduate/resampled_classification_dataset/augumentation"
config["resample_dir"] = "/home/liubo/data/graduate/resampled_classification_dataset/resample"


# 每个图片是由多个小图片连在一起的，这个函数把小图像分割出来存进列表并返回
def load_cube_img(src_path, rows, cols, size):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert rows * size == img.shape[0]
    assert cols * size == img.shape[1]
    res = np.zeros((rows * cols, size, size))
    img_height = size   # 48
    img_width = size    # 48
    for row in range(rows):    # 6
        for col in range(cols):  # 8
            src_y = row * img_height
            src_x = col * img_width
            # res[0] = img[0:48,0:48], res[1] = img[0:48, 48:96], res[7] = [0:48, 336:384]
            res[row * cols + col] = img[src_y:src_y + img_height, src_x:src_x + img_width]
    return res

def save_to_pic(des_path,rows,cols,size,cube):
    img = np.zeros((rows*size,cols*size))
    for row in range(rows):
        for col in range(cols):
            src_y = row * size
            src_x = col * size
            img[src_y:src_y+size,src_x:src_x+size] = cube[row*col+col]
    cv2.imwrite(des_path, img)



def augumentation():
    # 加载数据
    total_dir = config["total_dir"]
    zero_samples = glob.glob(total_dir+'/zero/' + '*.png')
    one_samples = glob.glob(total_dir + '/one/' + "*.png")        
    two_samples = glob.glob(total_dir + '/two/' + "*.png")
    three_samples = glob.glob(total_dir + '/three/' + "*.png")
    four_samples = glob.glob(total_dir + '/four/' + "*.png")

    origin_path = [zero_samples,
                   one_samples,
                   two_samples,
                   three_samples,
                   four_samples]

    # 进行数据增强
    class_dir_name_list = ["zero","one","two","three","four"]
    augumentation_dir = config["augumentation_dir"]
    for i in range(len(origin_path)):
        source_path_file_list = origin_path[i]
        dest_path = augumentation_dir + "/" + class_dir_name_list[i]
        for j in tqdm(range(len(source_path_file_list)),desc="augumentation processing (dir " + class_dir_name_list[i] + ")"):
            source_file = source_path_file_list[j]
            source_file_name = source_file.split("/")[-1].split(".")[0]
            suffix = ".png"
            row = 8
            col = 8
            size = 64
            origin = load_cube_img(source_file, 8, 8, 64) 
            filp_lr = origin[:,:,::-1]
            filp_ud = origin[:,::-1,:]
            origin_des_path = dest_path + "/" + source_file_name + "_ori" + suffix
            lr_des_path = dest_path + "/" + source_file_name + "_lr" + suffix
            ud_des_path = dest_path + "/" + source_file_name + "_ud" + suffix
            save_to_pic(origin_des_path,8,8,64,origin)
            save_to_pic(lr_des_path,8,8,64,filp_lr)
            save_to_pic(ud_des_path,8,8,64,filp_ud)

def resample():
    """
    凑齐 5折交叉验证 2000张 * 5类 * 5折
    """
    resample_dir = config["resample_dir"]
    augumentation_dir = config["augumentation_dir"] 
    class_dir_name_list = ["zero","one","two","three","four"]
    fold_name_list = ["fold0","fold1","fold2","fold3","fold4"]

    for i in range(len(fold_name_list)):
        for j in range(len(class_dir_name_list)):
            source_path = augumentation_dir + "/" + class_dir_name_list[i] + "/" +"*.png"
            source_path_file_list = glob.glob(source_path)
            sample_number = [random.randint(0,len(source_path_file_list)-1) for n in range(2000)]
            target_dir = resample_dir + "/" + fold_name_list[i] +"/"+class_dir_name_list[j]
            for k in tqdm(range(len(sample_number)),desc= "copy to " + target_dir):
                augumentation_name = source_path_file_list[sample_number[k]].split("/")[-1]
                resample_name = augumentation_name.split(".")[0] + "_%04d" % k + ".png"
                shutil.copy(source_path_file_list[sample_number[k]],target_dir+"/"+resample_name)

        

if __name__ == "__main__":
    # augumentation()
    resample()
    

    
    