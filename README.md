# ECE2195

# Task 1 GridDehazeNet for Image Dehazing

## Subtask 1 Reproduce the codes of GridDehazeNet

  In this task, we reporduce the work of GridDehazeNet[1]. The codes and the results are in the GridDehazeNet repo based on the source codes from https://github.com/proteus1991/GridDehazeNet. Following the instructions of the source codes, we first download the ITS (for indoor) training dataset from RESIDE. Then, copy hazy and clear folders from downloaded ITS to ./data/train/indoor/. The testing SOTS dataset is already given from the original source codes. Our repo provides some examples of the hazed and clear images.
  
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

 In [2], Zero Component Analysis (ZCA) has been proposed to preprocess images before input data entering the CNN model while the mean normalization and standardization filter serves as the baseline for comparison with different pre-processing techniques. In this part, we reproduce the experiment result of a 10-class image classigication task with three kinds of preprocess methods mentioned in [2], including mean normalizaion, standardization  and Zero Component Analysis(ZCA). Please see https://github.com/kuntalkumarpal/Preprocessing-Image-Classification-CNN.git for the original code. For mean normalizaion and standardization cases, we further applied RGB-normalization based on their code, otherwise the accuracy would be very low about 10%.

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
1.   
  
  
  
  
  
  
  
  
# Reference
[1] Liu, Xiaohong, et al. "Griddehazenet: Attention-based multi-scale network for image dehazing." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.<br>
[2] K. K. Pal and K. S. Sudeep, "Preprocessing for image classification by convolutional neural networks," 2016 IEEE International Conference on Recent Trends in Electronics, Information & Communication Technology (RTEICT), 2016, pp. 1778-1781.<br>
