# Drawing the Same Bounding Box Twice? Coping Noisy Annotations in Object Detection with Repeated Labels

This repository contains the code associated with the publication "Drawing the Same Bounding Box Twice? Coping Noisy 
Annotations in Object Detection with Repeated Labels" published at GCPR23 [[PDF](https://arxiv.org/abs/2309.09742)]. It contains (1) the extended mmdetection code with the methods
to aggregate ground truth approximations from noisy labels, (2) builders and pipeline elements to add necessary 
functionalities to mmdetection, (3) updated datasets modules to allow using the different noisy label data and (4) all 
configurations to reproduce the results presented in the paper. It is not possible to reproduce the aggregation done on 
the TexBiG test data, since these annotations are not publicly available and are only part of the leaderboard as shown below.

## Related Projects

The [predecessor](https://webis.de/downloads/publications/papers/tschirschwitz_2022.pdf) of this paper, presents an evaluation method for noisy labels in object detection and instance segmentation
as well as the training and validation dataset used here:

    @inproceedings{10.1007/978-3-031-16788-1_22,
      title={A Dataset for Analysing Complex Document Layouts in the Digital Humanities and its Evaluation with Krippendorff ’s Alpha},
      author={Tschirschwitz, David and Klemstein, Franziska and Stein, Benno and Rodehorst, Volker},
      booktitle="Pattern Recognition",
      year={2022},
      publisher="Springer International Publishing",
      address="Cham",
      pages="354--374",
    }

## Evaluation Server

The evaluation server can be found on [eval-ai](https://eval.ai/web/challenges/challenge-page/2078/overview).

## Installation

This example installation shows how to install the repository on Ubuntu using anaconda, an editable install of the
repository and installing PyTorch with GPU support.

1. Create an environment: `conda create --name gtiod python=3.8`
2. Activate environment: `conda activate gtiod`
3. Install [PyTorch 12.1](https://pytorch.org/get-started/previous-versions/#v1121): `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge` (Change the cuda version according to your systems requirements.)
4. Install mim: `pip install -U openmim==0.2.1`
5. Install mmcv: `mim install mmcv-full==1.6.1`
6. Install mmdet: `pip install mmdet==2.25.1`
7. Download the repository `git clone https://github.com/Madave94/gtiod.git`
8. Move into directory: `cd gtiod`
9. Install repository in editable mode: `pip install -e .`

If you encounter any issue, try to install a different version of mim, mmcv and mmdet. You should keep the PyTorch version
as it was the version used to create this code.

## Dataset Preparation

Downloading the TexBiG dataset is possible via [kaggle](https://kaggle.com/datasets/a1cb83673ee6d5fccabe1d8dfc9d0e01714ec0ff62d6b655bcfc5565d6380d97),
which contains a version of the dataset that is already prepared for immediate usage. Alternatively, there is also another 
version on [zenodo](https://zenodo.org/record/6885144), that would require to add the original test data to the training
data. Since in this publication new test data are introduced the old dataset can be used fully for training.

The VinDr-CXR data used, are part of a [challenge](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data)
that ran from December 2020 to March 2021. The images needs to be converted from dicom to png: 

    #!/usr/bin/env bash
    
    FILENAMES_TRAIN=$(ls "train_dicom/" | cut -d . -f 1)
    
    for FILENAME in $FILENAMES_TRAIN
    do
        convert train_dicom/$FILENAME.dicom -format png train/$FILENAME.png 
    done
    
    FILENAMES_TEST=$(ls "test_dicom/" | cut -d . -f 1)
    
    for FILENAME in $FILENAMES_TEST
    do
        convert test_dicom/$FILENAME.dicom -format png test/$FILENAME.png 
    done

To use the annotations, run the CLI scripts in ` python gtiod/utils/dataset_preperation --help` and `python gtiod/utils/dataset_split --help` to prepare 
the data. This will require you to provide the correct paths as command line argument which you can see when running the two above commands.

## Running the Code

Run `python tools/train.py configs/texbig/texbig_detectoRS_train_laem_intersection_val_laem_union.py /datasets/texbig_data --gpu 0`.
This would run the config as described in the config file using the data in `/datasets/texbig_data`. Change the path to the dataset
to where it is on your machine. Run the code from the project root folder.

The utils script `gtiod/utils/vindrcxr2kaggle.py`can be used to prepare coco results for submission to the challenge leaderboard,
effectively using it as an evaluation server.

To prepare a file for the TexBiG evaluation server use the `tools/inference.py` script. For example:

    python tools/inference.py \
    configs/texbig/texbig_detectoRS_train_laem_union_val_laem_averaging.py \ 
    logs/texbig_detectoRS_train_laem_union_val_laem_averaging/best_bbox_mAP_epoch_8.pth \ 
    /path/to/test/images \
    --results_path logs/texbig_detectoRS_train_laem_union_val_laem_averaging/detectoRS_results.json \ 
    --device cuda:0 \
    --dlcv_format \
    --model_name DetectoRS_ResNet50 \ 
    --model_version iccv23_submission_train_laem_union_val_laem_avg

This command would use the specified checkpoint in the logs folder and the specified config to create a file in `.json`
file, ready for leaderboard submission.

## Cite us

    @inproceedings{tschirschwitz2023aggregation,
      title={Drawing the Same Bounding Box Twice? Coping Noisy Annotations in Object Detection with Repeated Labels},
      author={Tschirschwitz, David and Benz, Christian and Florek, Morris and Noerderhus, Henrik and Stein, Benno and Rodehorst, Volker},
      booktitle={Pattern Recognition: 45th DAGM German Conference, DAGM GCPR 2023, Heidelberg, Germany, September 19--22, 2023, Proceedings},
      year={2023},
      organization={Springer}
    }