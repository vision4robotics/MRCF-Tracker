# Disruptor-Aware Interval-Based Response Inconsistency for Correlation Filters in Real-Time Aerial Tracking

Matlab implementation of our Disruptor-Aware Interval-Based Response Inconsistency for Correlation Filters in Real-Time Aerial Tracking (IBRI) tracker.

| **Test passed**                                              |
| ------------------------------------------------------------ |
| [![matlab-2017b](https://img.shields.io/badge/matlab-2017b-yellow.svg)](https://www.mathworks.com/products/matlab.html)|


# Abstract 
>Aerial object tracking approaches based on discriminative correlation filter (DCF) have attracted wide attention in the tracking community due to their impressive progress recently. Many studies introduce temporal regularization into the DCF-based framework to achieve a more robust appearance model and further enhance the tracking performance. However, existing temporal regularization approaches usually utilize the information of two consecutive frames, which are not robust enough due to limited information. Although some methods attempt to incorporate abundant training samples and generally improve the tracking performance, these improvements are at the expense of significantly increased computing consumption. Besides, most existing methods introduce historical information directly without denoising, which means background noises are also introduced into the filter training and may degrade the tracking accuracy. To tackle the drawbacks mentioned above, this work proposes a novel aerial object tracking approach to exploit disruptor-aware interval-based response inconsistency, i.e., IBRI tracker. The proposed method is able to incorporate historical interval information by utilizing responses in the filter training process, thereby obtaining a robust tracking performance while maintaining the real-time speed. Moreover, to reduce the disruptions caused by similar object, partial occlusion, and other challenging scenes, a novel disruptor-aware scheme based on response bucketing is introduced to detect the disruptor and enforce a spatial penalty for the disruptive area around the tracked object. Exhausted experiments on multiple well-known challenging aerial tracking benchmarks demonstrate the accuracy and robustness of the proposed IBRI tracker against other 35 state-of-the-art trackers. With a real-time speed of ~32 frames per second on a single CPU, the proposed approach can be applied for typical aerial platforms to achieve aerial visual object tracking efficiently.

# Publication and citation

IBRI tracker is proposed in our paper for IEEE TGRS 2020. Detailed explanation of our method can be found in the paper:

Changhong Fu, Junjie Ye, Juntao Xu,  Yujie He and Fuling Lin

Disruptor-Aware Interval-Based Response Inconsistency for Correlation Filters in Real-Time Aerial Tracking

# Contact 
Changhong Fu

Email: changhong.fu@tongji.edu.cn

Junjie Ye

Email: ye.jun.jie@tongji.edu.cn

# Demonstration running instructions

>Running demonstration of this tracker is very easy so long as you have MATLAB. Just download the package, extract it and follow two steps:
>
>1. Config seq name in `configSeqs_demo_for_IBRI.m`,
>
>2. Run `IBRI_Demo_single_seq.m`,
>   and the IBRI Demo should be running.

# Results on UAV datasets

### UAVDT

![](./results/UAVDT.png)

### UAV123@10fps

![](./results/UAV123@10fps.png)

### VisDrone2018-SOT

![](./results/VisDrone2018.png)



# Acknowledgements

We thank the contribution of `Hamed Kiani`, `Feng Li`, Dr. `Martin Danelljan` for their previous work BACF, STRCF and ECO. The feature extraction modules are borrowed from the ECO tracker (https://github.com/martin-danelljan/ECO) and STRCF tracker (https://github.com/lifeng9472/STRCF) and some of the parameter settings and functions are borrowed from BACF (www.hamedkiani.com/bacf.html) and STRCF.

