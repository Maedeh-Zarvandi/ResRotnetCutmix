## *Unsupervised Representation Learning by Predicting Image Rotations + Cutmix: Regularization Strategy to Train Strong Classifiers with Localizable Features*

### Introduction

The current code implements on [pytorch](http://pytorch.org/) the following ICLR2018 paper:    
**Title:**      "Unsupervised Representation Learning by Predicting Image Rotations"    
**Authors:**     Spyros Gidaris, Praveer Singh, Nikos Komodakis    
**Institution:** Universite Paris Est, Ecole des Ponts ParisTech    
**Code:**        https://github.com/gidariss/FeatureLearningRotNet   
**Link:**        https://openreview.net/forum?id=S1v4N2l0-

And ICCV2019 paper:
**Title:**      "Cutmix: Regularization Strategy to Train Strong Classifiers with Localizable Features"  
**Authors:**     Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo   
**Institution:** Clova AI Research, NAVER Corp., Clova AI Research, LINE Plus Corp., Yonsei University  
**Code:**        https://github.com/clovaai/CutMix-PyTorch  


**Abstrac (1st paper):**  
Over the last years, deep convolutional neural networks (ConvNets) have transformed the field of computer vision thanks to their  unparalleled capacity to learn high level semantic image features. However, in order to successfully learn those features, they usually require massive amounts of manually labeled data, which is both expensive and impractical to scale. Therefore, unsupervised semantic feature learning, i.e., learning without requiring manual annotation effort, is of crucial importance in order to successfully harvest the vast amount of visual data that are available today. In our work we propose to learn image features by training ConvNets to recognize the 2d rotation that is applied to the image that it gets as input.  We demonstrate both qualitatively and quantitatively that this apparently simple task actually provides a very powerful supervisory signal for semantic feature learning.  We exhaustively evaluate our method in various unsupervised feature learning benchmarks and we exhibit in all of them state-of-the-art performance. Specifically, our results on those benchmarks demonstrate dramatic improvements w.r.t. prior state-of-the-art approaches in unsupervised representation learning and thus significantly close the gap with supervised feature learning. For instance, in PASCAL VOC 2007 detection task our unsupervised pre-trained AlexNet model achieves the state-of-the-art (among unsupervised methods) mAP of 54.4%$that is only 2.4 points lower from the supervised case.  We get similarly striking results when we transfer our unsupervised learned features on various other tasks, such as ImageNet classification, PASCAL classification, PASCAL segmentation, and CIFAR-10 classification.

**Abstract (2nd paper):**  
Regional dropout strategies have been proposed to enhance the performance of convolutional neural network classifiers. They have proved to be effective for guiding the model to attend on less discriminative parts of objects (e.g. leg as opposed to head of a person), thereby letting the network generalize better and have better object localization capabilities. On the other hand, current methods for regional dropout remove informative pixels on training images by overlaying a patch of either black pixels or random noise. Such removal is not desirable because it leads to information loss and inefficiency during training. We therefore propose the CutMix augmentation strategy: patches are cut and pasted among training images where the ground truth labels are also mixed proportionally to the area of the patches. By making efficient use of training pixels and retaining the regularization effect of regional dropout, CutMix consistently outperforms the state-of-the-art augmentation strategies on CIFAR and ImageNet classification tasks, as well as on the ImageNet weakly-supervised localization task. Moreover, unlike previous augmentation methods, our CutMix-trained ImageNet classifier, when used as a pretrained model, results in consistent performance gains in Pascal detection and MS-COCO image captioning benchmarks. We also show that CutMix improves the model robustness against input corruptions and its out-of-distribution detection performances.


### Requirements
pytorch version 0.2.0_4 and higher

<!-- ### License
This code is released under the MIT License (refer to the LICENSE file for details).  -->

### Before running the experiments
* Inside the *FeatureLearningRotNet* directory with the downloaded code you must create a directory named *experiments* where the experiments-related data will be stored: `mkdir experiments`.
* You must download the desired datasets and set in [dataloader.py] the paths to where the datasets reside in your machine. We recommend creating a *datasets* directory `mkdir datasets` and placing the downloaded datasets there. 
* Note that all the experiment configuration files are placed in the [./config] directory.

### CIFAR-10 experiments
* In order to train (in an unsupervised way) the RotNet model on the CIFAR-10 training images and then evaluate object classifiers on top of the RotNet-based learned features see the [run_cifar10_based_unsupervised_experiments.sh] script. Pre-trained model (in pytorch format) is provided.
* In order to run the semi-supervised experiments on CIFAR-10 see the [run_cifar10_semi_supervised_experiments.sh]script.

###In order to:
* just apply rotations you need to use dataloader.py file and algorithm folder.
* first apply rotations then cutmix you need to use dataloader-rc.py file and algorithm-rc folder.
* first apply cutmix then rotations you need to use dataloader-cr.py file and algorithm-cr folder.
* have different images for each angle you need to use dataloader-rnb.py file and algorithm folder.

<!-- ### ImageNet and Places205 experiments
* In order to train (in an unsupervised way) a RotNet model with AlexNet-like architecture on the **ImageNet** training images and then evaluate object classifiers (for the ImageNet and Places205 classification tasks) on top of the RotNet-based learned features see the [run_imagenet_based_unsupervised_feature_experiments.sh](https://github.com/gidariss/FeatureLearningRotNet/blob/master/run_imagenet_based_unsupervised_feature_experiments.sh) script.
* In order to train (in an unsupervised way) a RotNet model with AlexNet-like architecture on the **Places205** training images and then evaluate object classifiers (for the ImageNet and Places205 classification tasks) on top of the RotNet-based learned features see the [run_places205_based_unsupervised_feature_experiments.sh](https://github.com/gidariss/FeatureLearningRotNet/blob/master/run_places205_based_unsupervised_feature_experiments.sh) scritp. -->


<!-- ### Download the already trained RotNet model
* In order to download the RotNet model (with AlexNet architecture) trained on the ImageNet training images using the current code, go to: [ImageNet_RotNet_AlexNet_pytorch](https://mega.nz/#!n81AnC6L!xTbo_D3xd7QOpOSG1UFSChmDr8mbcuWbVjhQMaC4yoE). Note that:   
  1. The model is saved in pytorch format.   
  2. It is not the same as the one used in the paper and probably will give (slightly) different outcomes (in the ImageNet and Places205 classification tasks that it was tested it gave better results than the paper's model).    
  3. It expects RGB images that their pixel values are normalized with the following mean RGB values `mean_rgb = [0.485, 0.456, 0.406]` and std RGB values `std_rgb = [0.229, 0.224, 0.225]`. Prior to normalization the range of the image values must be [0.0, 1.0]. -->


<!--  * In order to download the RotNet model (with AlexNet architecture) trained on the ImageNet training images using the current code and convered in caffe format, go to: [ImageNet_RotNet_AlexNet_caffe](https://mega.nz/#!ekVRlLJC!N23AlTHuGwJF87sS6f7QjUyGfVFllEOFVgKtcrvZvYk). Note that:   
   1. The model is saved in caffe format.  
   2. It is not the same as the one used in the paper and probably will give (slightly) different outcomes (in the PASCAL segmentation task it gives slightly better results than the paper's model).   
   3. It expects BGR images that their pixel values are mean normalized with the following mean BGR values `mean_bgr = [0.406*255.0, 0.456*255.0, 0.485*255.0]`. Prior to normalization the range of the image values must be [0.0, 255.0].   
   4. The weights of the model are rescaled with the approach of [Kraehenbuehl et al, ICLR 2016](https://github.com/philkr/magic_init).       -->
   

