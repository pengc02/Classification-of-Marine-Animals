# 大作业2——海洋生物分类

*by pengcheng 2020011075*

## 1. 环境配置

配置conda环境`conda create -n env python=3.8`
配置相关依赖`pip install -r requirements.txt`

## 2. 数据和模型

从清华云盘下载模型和数据：https://cloud.tsinghua.edu.cn/d/7b83454295e14ef6833f/

分别解压缩到`./database`和`./modelzoo`(根目录为main函数的目录)

## 3. 运行代码

Linux下：

训练:`bash train.sh`
测试: `bash test.sh`
可以在bash脚本中根据提示修改模型名称

绘制热图部分的说明参见`./gradcam/README.md`

## 3. 查看结果

您可利用wandb查看训练过程曲线。（注意需要在main函数中修改为你的用户名）

test模式下将会在终端直接打印测试结果，并完成混淆矩阵的可视化。

最终结果如下：

| model                    | acc  | precision | recall | f1    |
| ------------------------ | ---- | --------- | ------ | ----- |
| MyNet                    | 0.41 | 0.526     | 0.522  | 0.521 |
| MyNetPro                 | 0.54 | 0.586     | 0.757  | 0.658 |
| VGG16                    | 0.59 | 0.641     | 0.676  | 0.644 |
| VGG16-pretrain           | 0.68 | 0.663     | 0.868  | 0.748 |
| ResNet34                 | 0.58 | 0.587     | 0.798  | 0.671 |
| ResNet34-pretrain        | 0.69 | 0.68      | 0.903  | 0.774 |
| ResNet34-pretrain-frozen | 0.76 | 0.76      | 0.941  | 0.831 |