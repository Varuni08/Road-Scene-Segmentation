# Road Scene Segmentation using SegFormer

A semantic segmentation pipeline for urban road footage using a pretrained SegFormer model fine-tuned on the Cityscapes dataset.

## Features

* Processes road and driving videos frame by frame
* Performs pixel-level semantic segmentation
* Identifies roads, sidewalks, vehicles, pedestrians, buildings, vegetation and other urban objects
* Generates colour-coded segmentation masks
* Blends segmentation results with the original video frames
* Saves the processed output video locally

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* SegFormer
* OpenCV
* NumPy
* Pillow

## Model

This project uses the pretrained model:

`nvidia/segformer-b0-finetuned-cityscapes-512-1024`

## Installation

Install the required packages:

```
pip install -r requirements.txt
```

## Usage

Place your input video inside the project folder and update the video path in `road_segmentation.py`.

Run the script:

```
python road_segmentation.py
```

The segmented output video will be saved locally.

## Applications

* Autonomous driving research
* Road-scene understanding
* Urban mapping
* Traffic monitoring
* Computer vision experimentation
