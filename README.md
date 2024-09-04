# FIN



## [FIN: Flow-based Robust Watermarking with Invertible Noise Layer for Black-box Distortions](https://ojs.aaai.org/index.php/AAAI/article/view/25633)



Han Fang, Yupeng Qiu, Kejiang Chen*, Jiyi Zhang, Weiming Zhang, and Ee-Chien Chang*.

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

In this project we use DIV2K as the training dataset(which contains 800 images) and the validing dataset (which contians 100 images).

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

### White-box Noise Layers

For all white-box noise layers, we directly use the publicly available code from the MBRS repository. You can find the implementation here: [MBRS Noise Layers](https://github.com/jzyustc/MBRS/tree/main/network/noise_layers).



****


### Train
You will need to install the requirements, and change the settings in `config.py`, then run :

```bash
python train.py
```

The log files and experiment result information will be saved in `logging` in .txt format.
****

#### Some tips for training:

During the initial training phase, the Invertible Neural Network may exhibit instability. Based on extensive experimentation, we offer the following recommendations to help stabilize the model during this critical early stage:

1. Use a smaller learning rate: We suggest setting the learning rate to lr = 1e-4 for the first 15 epochs.

2. Increase the weight of the message loss: We recommend setting message_weight to 10,000 and stego_weight to 1 for the first 15 epochs.

3. If these settings still result in the loss becoming NaN, consider restarting the training process.

After successfully navigating the initial 15 epochs, you can adjust the message_weight and stego_weight according to your specific needs for robustness or visual quality.

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

### Combined Distortions

During training, the Combined Noise layer includes the following components:

- **JpegSS**: with quality factor Q=50
- **JpegTest**: with quality factor Q=50
- **Gaussian Blur**: with sigma=2.0, kernel size=7
- **Median Blur**:  with kernel size=7
- **Gaussian Noise**: with variance=0.05 and mean=0
- **Salt & Pepper Noise**: with probability=0.05
- **Dropout**: with probability=0.4
- **Cropout**: with height_ratio=0.7 and width_ratio=0.7

#### Benchmark comparisons on invisibility and robustness against combined noise.

| **Method** | VQ (dB) | Jpeg Compression (%) | S&P Noise (%) | Gaussian Noise (%)| Cropout (%) |
|------------|---------|-----------------------|-------------------|----------------------------|---------|
| FIN        | 41.72   | 97.07                 | 99.90             | 94.87                      | 89.26   |
| **Method** | **VQ (dB)** |**Dropout (%)**|**Gaussian Blur (%)**| **Median Blur (%)** |**Ave (%)** |
| FIN        | 41.72   | 99.90                | 99.90             | 98.73                      | 97.09   |










Contact: [qiuyupeng1999@gmail.com](mailto:qiuyupeng1999@gmail.com)

