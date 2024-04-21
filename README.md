# PyTorch Implementation of Poisson-Gaussian Holographic Phase Retrieval with Score-based Image Prior
by Zongyu Li, Jason Hu, Xiaojian Xu, Liyue Shen and Jeff Fessler.
Paper link: [To be added]

## Description
This study focuses on holographic phase retrieval in situations where the measurements are degraded by a combination of Poisson and Gaussian noise, 
as commonly occurs in optical imaging systems. 
We propose a new algorithm called “AWFS" that uses accelerated Wirtinger flow (AWF) with a learned score function as a generative prior.
Specifically, we formulate the PR problem as an optimization problem that incorporates both data fidelity and regularization terms. 
We calculate the gradient of the log-likelihood function for PR and determine its corresponding Lipschitz constant. 
Additionally, we introduce a generative prior in our regularization framework by using score matching to capture information about the gradient of image prior distributions.

## Getting Started

### Dependencies

Run `pip -r install requirements.txt`.

### Installing

* Download data from [data](https://drive.google.com/drive/folders/1k2RfVD1Yg-JNu4B__Ttmd0kSIFKZsd65?usp=sharing).
* Download pretrained models from [pnp models](https://drive.google.com/drive/folders/1gDYgz5iaEOCCQB6A9v0Fj9SMwEDoh2Ji?usp=sharing), [ddpm models](https://drive.google.com/drive/folders/1AjYxVa0wjv0VP2iL0f46UyMH3SO-pZUm?usp=sharing) and [score models](https://drive.google.com/drive/folders/1GcFlXxHcvIy4ldfZYSSgEbyxvbJOMRyE?usp=sharing).
* After downloading, you will have a directory of following structure:
```plaintext
2023-PGPR/
│
├── config_purple/      # configuration files
├── config_celebA/     
├── config_density/
│
├── src/                # Source code files
│   ├── main.py         # Main application entry point
│   ├── utils.py        # Helper functions
│   └── ......
│            
│
├── pnpmodels/              # pretrained checkpoints
├── scoremodels/
├── ddpmmodels/
│
├── data/               # Data files
│   ├── purpletest.mat
│   ├── density_small_test.mat
│   └── celebA_small_test.mat
│
├── .gitignore          # Specifies intentionally untracked files to ignore
├── LICENSE             # License file
├── README.md           # The README file
└── requirements.txt    # Required dependencies
```

### Set up configuration files

#### Settings

- `gpu_ids`: A string that specifies the GPU IDs to be used for the experiments, where `"0"` denotes the first GPU.

#### Experiment Arguments (`expargs`)
- `savedir`: Directory where the results will be saved.
- `datadir`: Directory where the data is located.
- `dataset_name`: The name of the dataset being used for the model, such as `"virusimg"`.
- `init`: Initialisation method used in the experiments.
- `img_to_do`: List of image indices to process. An empty list `[]` means all images are processed.
- `exp_to_do`: List of algorithms to be conducted. An empty list `[]` means all algorithms will be ran.
- `ncore`: Number of CPU cores to be utilized.
- `imgsize`: Size of the images.
- `scaleSYS`: System scale parameter.
- `sigma`: Standard deviation of Gaussian noise.
- `delta`: Parameter for approximating the infinite summation in PG log-likelihood.
- `regTV`: Parameter for the total vairation regularization.
- Various other `niter` parameters that specify the number of iterations for different algorithms.
- Scale and `rho` parameters for different PnP algorithms when combining with the CNN denoised results.

#### Methods (Denoise)
Including model names and paths to pre-trained models for different noise levels.
- `natureimg`, `virusimg`, `celebA_small`, `density_small`: Different datasets or image types to apply denoising on.
  - `dnn_name`: The name of the denoising neural network, e.g., `"dncnn"`.
  - `sgm_name`: Signature for noise model, e.g., `"sgm15"`.
  - `model_path`: Object containing paths to the trained model files for different levels of noise.

#### Networks

Specifications of the DnCNN (UNet) architecture for use in the experiments.

- `dncnn`: Denoising Convolutional Neural Network parameters.
  - `dimension`: The dimensionality of the input data.
  - `ic`: Number of input channels.
  - `oc`: Number of output channels.
  - `depth`: The depth (number of layers) of the network.
  - `kernel_size`: The size of the convolution kernels.
  - `features`: Number of features at each layer.
  - `groups`: The number of blocked connections from input channels to output channels.
  - `is_bn`: Boolean indicating if batch normalization is used.
  - `is_sn`: Boolean indicating if spectral normalization is used.
  - `is_res`: Boolean indicating if residual connections are used.
- `unet`: Parameters for Unet architecture.
  - `dim`: Dimensionality specification for Unet.


## Executing program
* After making changes to the configuration file, you can do:
```
cd src && python main.py
```
