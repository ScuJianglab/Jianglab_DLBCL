# Single-Cell Profiling Identifies CD4+CXCR5+PD-1- Tfh Cells 1 as Predictive Biomarkers for R-CHOP Therapy Response in human DLBCL
### Sisi Yu1,5, †, Hao Kong1, †, Huaichao Luo1,6, †,Xingzhong Zhao7,†, Guanghui Zhu8,†, Jie Yang1, Xiangji Wu1, Yu Dai1, Chunwei Wu1, James Q. Wang 7, Dan Cao4, §, Yang Xu3, §, Hong Jiang1,2, § , Ping Wang1, §,#


## Requirements
Clone this repo then install the dependencies using:
```bash
conda env create -f environment.yml
```
Note that we used `pytorch==1.12.1` and a RTX4090 GPU for all experiments.
## Data Preparation
For training the model, we need the paried images for DAPI and mIHC images. Then you can run the script 
```python
python WSINorm/wsi_registration.py
```
change the path to generate the patch image of DAPI images and IHC images. 
```python
data_dir = '/homellm8t/zhaoxz/' ## root dir

other_modal = '/homellm8t/zhaoxz/HE_merge_small' # target image
patch_size = 1024
fmt = 'tiff'
num_processes = 16

source = os.path.join(data_dir, 'IHE_small_regis') #source image 
target = os.path.join(data_dir, other_modal)

save_dir = os.path.join(data_dir, 'samll_HE/masked_patch') ## save dir
```
Then you will get the path images in save_dir, then you run the script to split images into train and val dataset 
```python
python segmentation_dataset.py
```

## Training from Scratch
We use `experiments/mist_launcher.py` to generate the command line arguments for training and testing. More details on the parameters used in training our models can be found in that launcher file.

To train a model from scratch on the MIST dataset:
```bash
python -m experiments mist train 0
```

## Testing and Evaluation
Again the same `mist_launcher.py` can be used for testing:
```bash
python -m experiments mist test 0
```

We provide our pretrained models on our private dataset.
The weights can be downloaded from [Google Drive](https://drive.google.com/open?id=1xdiaoDmC-rfEwmRya1Yf8eTWi6l4ez3e&usp=drive_copy).
To use one of the provided models for testing, modify the `name` and `checkpoints` arguments in the launch option. 
For example, to use `Our_dapi_mIHC_lambda_linear`:
```
name="Our_dapi_mIHC_lambda_top",
checkpoints_dir='/path/to/pretrained/'
```

The evaluation code that was used to generated the results in the paper is provided in `evaluate.py`.

## Acknowledgement

This repo is built upon [Adaptive Supervised PatchNCE Loss](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE/tree/master).
