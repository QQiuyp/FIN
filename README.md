# FIN



## FIN: Flow-based Robust Watermarking with Invertible Noise Layer for Black-box Distortions



Han Fang, Yupeng Qiu, Kejiang Chen*, Jiyi Zhang, Weiming Zhang, and Ee-Chine Chang*.

> This is the source code of paper FIN: Flow-based Robust Watermarking with Invertible Noise Layer for Black-box Distortions, which is received by AAAI' 23.


****

### Requirements

The core packages we use in the project and their version information are as follows:

- kornia `0.6.6`
- natsort `7.1.1`
- numpy `1.22.3`
- pandas `1.4.3`
- torch `1.12.0`
- torchvision `0.13.0`

****

### Dataset

In this project we use DIV2K as the training dataset(which contains 800 images) and the testing dataset (which contians 100 images).

The data directory has the following structure:
```
├── Datasets
│   ├── DIV2K_train
│   │   ├── 0001.png
│   │   ├── ...
│   ├── DIV2K_valid
│   │   ├── 0801.png
│   │   ├── ...
├── 

```


****


### Train
You will need to install the requirements, and change the settings in `config.py`, then run :

```bash
python train.py
```

The log files and experiment result information will be saved in `logging` in .txt format.
****



### Test

Since the black-box noise needs to be added by the user, the part of the test is divided into the message embedding part and the message extracting part. These two parts are implemented by `encode.py` and `decode.py` respectively.

#### Message Embedding Part
```bash
python encode.py
```


#### Message Extracting Part
```bash
python decode.py
```

There are some parameters for `encoode.py` and `decode.py`. Use
```bash
python encode.py(decode.py) --help
```
to see the description of all of the parameters.
****




Contact: [qiu_yupeng@u.nus.edu](mailto:qiu_yupeng@u.nus.edu)

