
# How to use it


## step 1 data process


`python preprocess.py`

you can get the data like this,each npy file shape is  `160*160*4`,use each slice below and above as input image,the mid slice as mask(`160*160`)

```
data---
    trainImage---
        BraTS19_Training_001_0.npy
        BraTS19_Training_001_1.npy
        ......
    trainMask---
        BraTS19_Training_001_0.npy
        BraTS19_Training_001_1.npy
        ......
    testImage---
        BraTS19_Testing_001_0.npy
        BraTS19_Testing_001_1.npy
        ......
    testMask---
        BraTS19_Testing_001_0.npy
        BraTS19_Testing_001_1.npy
        ......
```

## step 2 Train the model


`python train.py`

## step 3 Test the model


`python test.py`

you can download the models from : [BaiduNetdisk](https://pan.baidu.com/s/1JtZ1r7M2SFMnSvEz9IcQIw) code: 3jdw
