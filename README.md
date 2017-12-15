# GLAD
This repository provides the code for our ACM MM17 paper [GLAD: Global-Local-Alignment Descriptor for Pedestrian Retrieval](https://arxiv.org/pdf/1709.04329.pdf)

![](https://github.com/JoinWei-PKU/GLAD/blob/master/framework.png)

### Step.1 Pose Estimation
The first stage is to estimate the human keypoints.
We used the deepercut model provided in [DeeperCut](https://github.com/eldar/deepcut). Especially, we utilize the single person pose estimation model provided by the authors.

Afer pose estimation, please detect the three parts according to our paper. Example image is as followes:
![](https://github.com/JoinWei-PKU/GLAD/blob/master/datasets/example2.jpg)

You can utilize any pose estimation methods to replace DeeperCut.

### Step.2 Descriptor Learning
### Make our caffe
   We have modify the original caffe, please make our provided caffe before run our code.
### Dataset
   Download [Market1501 Dataset](http://www.liangzheng.org/Project/project_reid.html). Then process these raw data as step.1.
### ImageNet Pretrained model
   Download [GoogLeNet model](https://github.com/lim0606/caffe-googlenet-bn) pretrained on Imagenet.
### Train our GLAD
   1. Modify the `prototxt\train_val.prototxt`. Please modify the dataset path in the file.
   2. End up training with 10,0000 iterations. More details, please see the `prototxt\solver_stepsize_6400_2_step3_ver4_65.prototxt`

### Step.3 Test 
   1. Extract fc6(and layer1/fc6, layer2/fc6, layer3/fc6) features.
   2. L1 normalization is needed.
   3. Adding weights for these four features according to our paper.

### Our Model
   1. If you require our trained model, please contact Longhui Wei(weilh2568@gmail.com). 
   2. If you have any questions about our code or paper, please contact Longhui Wei

### Citation
Please cite this paper in your publications if it helps your research:
```
@inproceedings{wei2017glad,
  title={GLAD: Global-Local-Alignment Descriptor for Pedestrian Retrieval},
  author={Wei, Longhui and Zhang, Shiliang and Yao, Hantao and Gao, Wen and Tian, Qi},
  booktitle={ACM MM},
  year={2017}
}
```
