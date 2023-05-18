# NUAA-计算机视觉期末大作业

### 题目

1. **低层视觉及传统方法：使用Matlab/Python/C等自己编程实现至少一个以下算法：（40分）**
   1. 图像共生直方图特征及其显著图检测。详见PAMI原论文S. Lu, C. Tan and J. -H. Lim, "Robust and Efficient Saliency Modeling from Image Co-Occurrence Histograms," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 1, pp. 195-201, Jan. 2014.
   2. 暗通道先验及图像去雾霾。详见PAMI原论文K. He, J. Sun and X. Tang, "Single Image Haze Removal Using Dark Channel Prior," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 12, pp. 2341-2353, Dec. 2011.
   3. 视频中人物去阴影。详见PAMI原论文A. Prati, R. Cucchiara, Detecting moving shadows: Algorithms and evaluation, TPAMI, 2003.R. Cucchiara, et. al, Detecting moving objects, ghosts, and shadows in video streams, TPAMI, 2003.
   4. 景建模/前景分割（结合课上讲解的方法）自己用手机拍摄照片或视频，得到实验可视化结果，讨论其性能。除了图像读写操作等常规图像操作外，算法核心部分不可直接使用他人开源代码。
2. **高层视觉及传统/深度学习方法：使用开源代码实现至少一组以下算法：（40分）**
   1. 单目标跟踪器：核相关滤波器KCF（传统方法），Siamese Network或其改进方法（深度网络）的对比实验。数据集比如VOT,PETS多目标跟踪器：FairMOT与任何一种多目标跟踪器的对比实验。数据集比如MOT17\19
   2. 行人检测器：HOG特征行人检测或其改进方法，与任何一种传统/深度行人检测器对比实验。数据集比如PETS人脸检测器:   如Tiny Face，或其他人脸检测方法，选任意两种的对比实验。数据集比如widerface/FDDB
   3. 语义分割：PSPNET/DeeplebV3+，或其他语义分割方法，选任意两种的对比实验。数据集比如CityScspe
   4. 其中任何一项计算机视觉任务的两种方法的同样实验设置下的对比实验。选择一个主题后，根据原论文选择任何一个公开数据集，完成实验对比，主要是定量实验对比，并适当给出几个可视化结果。不具备训练条件的同学，可以不训练深度模型，直接加载开源的网络权重文件完成实验分析。
3. **进阶：（20分）自选一个较新的计算机视觉任务（问题）及较新的方法的复现和讨论。**（最好是近两年发表在PAMI,TIP,TNNLS,TMM,CVPR,ICCV,ECCV,NIPS,ICML,ICLR,AAAI,IJCAI,。。。。等期刊会议的工作）
4. 其他要求：报告要求包含: 1课设简介，2关键代码(题2和3或许代码量过大，且非原创性内容，不需要将代码附在报告中，请给云盘链接或github链接做评分参考)、3实验设置（数据集使用细节、参数设置、实验过程等细节）、4可视化及定量实验结果展示及说明，5对实验结果（定性/定量）现象的原理性分析，6.结论与总结。三个题均以此结构来完成。篇幅不限。最后可以另附课程心得。雷同大作业将严重影响成绩，严重雷同或将取消成绩。可1-3人一组。多人请写清分工。原则上，多人组队要求更高。最后一周两次课将组织课设展示和讲解，包含0-20分加分。请于最后一次课结束当天为最终提交日期，将报告pdf（文件名注明姓名）发送至指定邮箱。

### 文件

- 1为大作业第一题
  - 实现何凯明去雾
  - 可视化效果及数据链接如下：
- 2为大作业第二题
  - 语义分割
  - 参考代码链接如下：
    - PspNet：
    - Deeplabv3+：
- 3为大作业第三题
  - OpenAI CLIP
  - 可视化效果及数据链接如下：