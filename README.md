# ECE2195

# Task 1 GridDehazeNet for Image Dehazing

## Subtask 1 Reproduce the codes of GridDehazeNet

  In this task, we reporduce the work of GridDehazeNet[1]. The codes and the results are in the GridDehazeNet repo based on the source codes from https://github.com/proteus1991/GridDehazeNet. Following the instructions of the source codes, we first download the ITS (for indoor) training dataset from RESIDE(https://paperswithcode.com/dataset/reside). Then, copy hazy and clear folders from downloaded ITS to ./data/train/indoor/. The testing SOTS dataset is already given from the original source codes. Our repo provides some examples of the hazed and clear images.
  
  ### Requirements :
```sh
* Python >= 3.7
* CUDA >= 10.2
* Pytorch >= 1.7.0
* Torchvision >= 0.8.0
* Torchsummary >= 1.5.1
* Numpy >= 1.21.2
* Scipy >= 1.7.1
```

  
 To train the dehazing model with the pre-processing module, we run the following instruction:
 
    nohup python3 train.py > train.out 2>&1 &
  
 The training result is shown in the 'train.out' file, and the checkpoint of the well-trained model is saved in 'indoor_haze_best_3_6_finetune'.
 
 To train the dehazing model without the pre-processing module, we run the following instruction:
 
    nohup python3 trainnopre.py > trainnopre.out 2>&1 &
  
 The training result is shown in the 'trainnopre.out' file, and the checkpoint of the well-trained model is saved in 'indoor_haze_best_3_6_finetune_nopre'.
 
 To validate the well-trained model with the pre-processing module, we run the following instruction:
 
    nohup python3 test_finetune.py > test_finetune.out 2>&1 &
  
 The testing result is shown in the 'test_finetune.out' file, the PSNR is 32.12, and the SSIM is 0.9833.
 
 To validate the well-trained model without the pre-processing module, we run the following instruction:
 
    nohup python3 test_finetune_nopre.py > test_finetune_nopre.out 2>&1 &
  
 The testing result is shown in the 'test_finetune_nopre' file, the PSNR is 30.03, and the SSIM is 0.9789.
 
 ## Subtask 2 Establish an analytical model for GridDehazeNet
 
   We establish an analytical model to implement GridDehazeNet on U200 FPGA board. The model and the results are in the FPGAmodelT1 repo.
   
   To receive the best selection of Tm and Tn for the whole network, the best of Tr, Tc for each Convolution layer, and the estimate latency for GridDehazeNet with the pre-processing module, we run the following instruction:
   
    nohup python3 modelfpga.py > modelfpga.out 2>&1 &
   
The result is shown in the 'modelfpga.out' file.

To receive the best selection of Tm and Tn for the whole network, the best of Tr, Tc for each Convolution layer, and the estimate latency for GridDehazeNet without the pre-processing module, we run the following instruction:

    nohup python3 modelfpganopre.py > modelfpganopre.out 2>&1 &
   
The result is shown in the modelfpganopre.out' file.
  
# Task 2 Preprocessing for Image Classification by CNN(ZCA)

## Subtask 1 Reproduce the codes of ZCA

 In [2], Zero Component Analysis (ZCA) has been proposed to preprocess images before input data entering the CNN model while the mean normalization and standardization filter serves as the baseline for comparison with different pre-processing techniques. In this part, we reproduce the experiment result of a 10-class image classification task with three kinds of preprocess methods mentioned in [2], including mean normalization, standardization  and Zero Component Analysis(ZCA). Please see https://github.com/kuntalkumarpal/Preprocessing-Image-Classification-CNN.git for the original code. For mean normalizaion and standardization cases, we applied RGB-normalization based on their code, otherwise the accuracy would be as low as about 10%(ZCA preprocess version already has RGB norm in their code).

### Requirements :
```sh
* Python 2.7
* matplotlib
* opencv-python==4.2.0.32
* theano==0.7
* NVIDIA drivers (If using GPU) 
* cifar-10-batches-py(See ./dataset/cifar-10-batches-py/readme.html)
```

### Workflow :
1. Download cifar-10 dataset from the link procided in ./ZCA/datasets/cifar-10-batches-py/readme.html and unzip it in ./ZCA/datasets/cifar-10-batches-py/.<br>
2. Reproduce Preprocess + CNN <br>
Here we use ./ZCA/RGB_ZCA as an exmple to illustrate how these code works.<br>

Preprocessing the input images<br>
```sh
cd ./ZCA/RGB_ZCA
python cifar_loader_ZCA.py
```

CNN_ZCA.py will call its corresponding loader(here loader_centerd_ZCA.py) to load the image after preprocessing, then it's ready to start training and testing<br>
```sh
THEANO_FLAGS='floatX=float32,device=gpu,gpuarray__preallocate=1'  python CNN_ZCA.py
```
### Exepected Result :
We set "fLayerOutNeuron"=200, "learnRate"=0.05, "lambda=10.0". After 20 epoach, we expected to get 67.06% test accuracy.<br>
![74d2321237dd85837b6abcba0a4323a](https://user-images.githubusercontent.com/77606152/165691093-6f585164-6f2f-42fa-b3aa-907e8468f61e.png)<br>

 ## Subtask 2 Establish an analytical model for ZCA
 
 As described in Task 1, we also implemented a Matlab version of analytical model. In this analytical model, we aim to find the overall optimized unrolling factor when depolying ZCA preprocessing(Acturally a Matrix-Multiply Kernel) and VGG16 network on U200 FPGA.<br>
![image](https://user-images.githubusercontent.com/77606152/165982405-586d5528-83da-4d53-a1b4-6bb0a56763e2.png)<br>
  
  
  
  
  
# Reference
[1] Liu, Xiaohong, et al. "Griddehazenet: Attention-based multi-scale network for image dehazing." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.<br>
[2] K. K. Pal and K. S. Sudeep, "Preprocessing for image classification by convolutional neural networks," 2016 IEEE International Conference on Recent Trends in Electronics, Information & Communication Technology (RTEICT), 2016, pp. 1778-1781.<br>
