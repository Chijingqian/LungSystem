# LungSystem
Lung cancer detection ,nodule dection and classification System using deep learning

## dataset

## LUNA 16
LUNA 16的数据集
- 官网 https://luna16.grand-challenge.org/
- 自己保存（百度云盘有保存）

## graduate_dataset

## 数据地址
202.118.232.8

H:\medical_data_for_huodong\Images\2008\20080504\3107

Dicom 影像文件：H:\medical_data_for_huodong\images

execl 诊断报告：H:\medical_data_for_huidong\knowledge_graph_data

jpeg 标签文件： H:\medical_data_for_huodong\taged_picture

代码文件：H:\medical_data_for_huodong\pycharm_code

- 一个人可能最多分成3套 Standard MAC Pet


### 数据理解
- PNG 无损格式，不能存放jpeg，有损

## 明月分割的数据集合统计
- 训练集

## 相关视频
[youtube视频](https://www.youtube.com/watch?v=Dhf6NOVQCjk)

## 相关文档
- [Grad-cam](https://arxiv.org/abs/1610.02391)
- [关于医疗影像的mhd和dcm格式图像的读取和坐标转换](https://blog.csdn.net/zyc2017/article/details/84030903)
- [dicom数据格式说明](https://www.dicomlibrary.com/dicom/)
- [opencv-python文档](https://opencv-python-tutroals.readthedocs.io/en/latest/)
- [itk-snap](http://www.itksnap.org/download/snap/process.php?link=11443&root=nitrc)可以对数据进行标注
- [dicom转换.nii.gz](https://www.jianshu.com/p/4a1a2675a61b)
- [uwsgi参考文档](https://uwsgi-docs.readthedocs.io/en/latest/WSGIquickstart.html)

## 参考论文
- [lung nodule Detection via Deep reinforcement learning]
- []


# 工作计划 
## 日志
- 2019年11月上旬整理了学习了deeplung的代码
- 2019年11月中旬学习了keras的使用方法
- 2019年11月下旬了解了自己的数据库，并且查看了明月的代码，完成了初版代码
- 2019年12月6日，尝试适应Django和一些JS将DICOM文件显示出来,没有尝试成功
- 2019年12月17日，意识到数据集上的问题，需要重新划分数据集
- 2019年12月23日，发现了之前我自己实现的deeplung存在缺陷，之前实现的是在python27上的
- 2019年12月28日，发现如果使用重采样的方式，过采样到 ***2000张 \* 5类 \* 5折交叉验证*** 可能很快就拟合到100%了，所以计划一部分过采样，一部分欠采样： ***200张 \* 5类 \* 5折交叉验证***
- 2019年12月30日，调查了一些数据增强的通用方法，简单读了 MIT的张宏毅的mixup方法 

## TODO
- 数据集重新划分
- tfrecord
- 新网络
- 进行可视化操作
- 检测数据重新提取
- boosting方法

