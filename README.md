# Cloud Based Machine Learning

This repo contains my work for the cloud based machine learning exercises.

Most of the work is in Jupyter notebooks. The repo also includes the datasets, screenshots, saved outputs, model weights, and a diary document that I used while doing the exercises.

I tried to keep each exercise in its own folder, so it is easier to follow.

## Repository structure

```text
.
|-- First_exercise/
|-- Second_exercise/
|-- Third_exercise/
|-- FourthA_exercise/
|-- FourthB_exercise/
|-- FourthC_exercise/
|-- FourthD_exercise/
|-- Cloud-Based Machine learning diary.docx
|-- README.md
`-- .gitignore
```

There is also a local `.venv` folder and a `.env` file in my working copy. They are local environment files, not the main project work.

## First exercise

Folder: `First_exercise`

This part is about building an image dataset from Pexels.

The notebook is:

```text
First_exercise/pexels_dataset.ipynb
```

What I did here:

1. Downloaded images from Pexels for three classes: `cat`, `dog`, and `car`.
2. Used CLIP to filter images that did not look close enough to the class name.
3. Removed duplicate or near duplicate images with image hashing.
4. Split the data into train and test folders.
5. Kept screenshots from Teachable Machine.
6. Saved a converted Keras model from Teachable Machine.

Main folders:

```text
First_exercise/pexels_dataset/
First_exercise/Split-data/
First_exercise/Saved-model/
First_exercise/Screenshots/
```

The current local image counts are:

```text
Raw Pexels images:
car: 97
cat: 97
dog: 95

Train/test split:
train/car: 77
train/cat: 78
train/dog: 76
test/car: 20
test/cat: 19
test/dog: 19
```

The saved model files are inside:

```text
First_exercise/Saved-model/converted_keras/
```

## Second exercise

Folder: `Second_exercise`

This exercise is about binary classification for breast ultrasound images from the BUSBRA dataset.

The notebook is:

```text
Second_exercise/classification.ipynb
```

The dataset folder contains:

```text
Second_exercise/BUSBRA/Images/
Second_exercise/BUSBRA/Masks/
Second_exercise/BUSBRA/bus_data.csv
Second_exercise/BUSBRA/5-fold-cv.csv
Second_exercise/BUSBRA/10-fold-cv.csv
```

The local dataset has:

```text
Images: 1875 png files
Masks: 1875 png files
```

What I did in the notebook:

1. Loaded the BUSBRA metadata and split files.
2. Used the bounding boxes from the metadata.
3. Cropped the ultrasound images around the lesion area.
4. Created train, validation, and test data.
5. Used basic augmentation like crop, horizontal flip, and small rotation.
6. Trained a small CNN from scratch.
7. Trained MobileNetV2 with ImageNet weights, first frozen and then partly fine tuned.
8. Picked the classification threshold from the validation set, not from the test set.
9. Evaluated both models on the test set.

The notebook output shows these test results:

```text
CNN:
accuracy: 0.8298
balanced accuracy: 0.8165
AUC: 0.8976

MobileNetV2:
accuracy: 0.8085
balanced accuracy: 0.8306
AUC: 0.9106
```

I do not treat these as final medical results. This is a course exercise on a fixed dataset split. The notebook is useful for comparing the two model approaches, but the result should be read carefully.

Screenshots from the notebook are stored in:

```text
Second_exercise/Screenshots/
```

## Third exercise

Folder: `Third_exercise`

This exercise continues with Pexels images, but this time for object detection with YOLO.

The notebooks are:

```text
Third_exercise/pexels_dataset.ipynb
Third_exercise/Annotating_Yolo.ipynb
```

What I did:

1. Built a Pexels dataset for `bicycle`, `laptop`, and `pizza`.
2. Used CLIP again to filter images.
3. Used YOLO-World to auto annotate the images.
4. Converted the labels into YOLO format.
5. Split the detection data into train and validation folders.
6. Trained YOLOv8n for 30 epochs.
7. Saved training and validation outputs under `runs`.

Current raw image counts:

```text
bicycle: 98
laptop: 99
pizza: 98
```

The first auto annotated YOLO dataset has:

```text
train images: 232
train labels: 232
val images: 58
val labels: 58
```

The repo also has two later YOLO dataset folders:

```text
Third_exercise/Pexels-dataset/yolo_dataset_A/
Third_exercise/Pexels-dataset/yolo_dataset_B/
```

Both currently contain:

```text
train images: 199
train labels: 199
val images: 50
val labels: 50
```

The YOLO training output is here:

```text
Third_exercise/runs/detect/yolo_runs/exp/
```

The notebook validation output for the trained YOLOv8n model shows:

```text
all classes:
precision: 0.941
recall: 0.929
mAP50: 0.970
mAP50-95: 0.863

bicycle mAP50-95: 0.797
laptop mAP50-95: 0.896
pizza mAP50-95: 0.896
```

There is one important limitation. The labels were created automatically, and the notebook notes that some boxes were not good. Some boxes covered too much of the image, especially for bicycle. So the result is useful for learning the full detection pipeline, but the labels are not perfect ground truth.

Model weight files in this folder include:

```text
yolov8n.pt
yolov8s-worldv2.pt
yolo26n.pt
```

## Fourth exercise A

Folder: `FourthA_exercise`

This part is about neural style transfer.

The notebook is:

```text
FourthA_exercise/Notebook.ipynb
```

What I did:

1. Loaded images from the third exercise Pexels dataset.
2. Used the TensorFlow Hub arbitrary image stylization model.
3. Used a Kandinsky image as the style image.
4. Tested the style transfer on two examples.
5. Stylized a small batch from the bicycle, laptop, and pizza classes.
6. Compared different style strengths by blending the original and stylized images.

Notebook screenshots are saved in:

```text
FourthA_exercise/notebook_output_screenshots/
```

There are 6 screenshot files there, including batch comparisons and style strength comparison.

## Fourth exercise B

Folder: `FourthB_exercise`

This part is about image colorization with DeOldify.

The notebook is:

```text
FourthB_exercise/notebook.ipynb
```

What I did:

1. Used the DeOldify project code from the `DeOldify` folder.
2. Loaded black and white input images from `input_images`.
3. Created stable colorized versions.
4. Created artistic colorized versions for comparison.
5. Saved the output images and notebook screenshots.

Main folders:

```text
FourthB_exercise/DeOldify/
FourthB_exercise/input_images/
FourthB_exercise/colorized_images/
FourthB_exercise/notebook_output_screenshots/
FourthB_exercise/result_images/
```

Current local files:

```text
input_images: 5 files
colorized_images: 11 files
notebook_output_screenshots: 14 files
result_images: 0 files
```

The `DeOldify` folder is a copied project folder with its own code, notebooks, requirements, and README.

## Fourth exercise C

Folder: `FourthC_exercise`

This part is about image generation and upscaling with diffusion models.

The notebook is:

```text
FourthC_exercise/notebook.ipynb
```

What I did:

1. Loaded Stable Diffusion 1.5 with `diffusers`.
2. Generated five 360 by 360 images from text prompts.
3. Made a 2 by 2 style comparison grid for the same mountain lake subject.
4. Loaded the Stable Diffusion x4 upscaler.
5. Upscaled the generated eagle image.

Output folders:

```text
FourthC_exercise/generated_images/
FourthC_exercise/upscaled_images/
```

Current local files:

```text
generated_images: 11 files
upscaled_images: 2 files
```

## Fourth exercise D

Folder: `FourthD_exercise`

This part is about the Learning to Paint project.

The notebook is:

```text
FourthD_exercise/notebook.ipynb
```

What I did:

1. Used the `LearningToPaint` project folder.
2. Downloaded or used the renderer and actor weights.
3. Patched `test.py` in the notebook because the original code was written for an older PyTorch loading behavior.
4. Prepared two input images: one photo and one public domain painting.
5. Ran the painting process for both inputs.
6. Saved generated frames under `runs`.
7. Created mp4 videos with ffmpeg.
8. Saved notebook screenshots for the start, middle, and end comparisons.

Main folders:

```text
FourthD_exercise/LearningToPaint/
FourthD_exercise/assertinputs/
FourthD_exercise/inputs/
FourthD_exercise/runs/
FourthD_exercise/videos/
FourthD_exercise/notebook_output_screenshots/
```

Current local files:

```text
inputs: 3 files
assertinputs: 2 files
videos: 3 mp4 files
notebook_output_screenshots: 5 files
```

The videos are:

```text
painting.mp4
photo.mp4
webcam.mp4
```

The `LearningToPaint` folder also contains `actor.pkl` and `renderer.pkl`.

## Root files

Important root files:

```text
README.md
.gitignore
Cloud-Based Machine learning diary.docx
```

The diary document is my written course diary.

The `.gitignore` is set up to ignore local environments, datasets, generated outputs, screenshots, documents, and large model files. Some of those files are still present in my local working copy because they are part of the exercises and outputs I used.

## How to run

There is no single `requirements.txt` for the whole repo.

The easiest way is to open the needed notebook and install the packages used in that notebook. The main packages used across the repo are:

```text
torch
torchvision
tensorflow
tensorflow_hub
diffusers
ultralytics
Pillow
opencv-python
imagehash
scikit-learn
pandas
matplotlib
requests
gdown
```

Some notebooks also need:

```text
OpenAI CLIP
DeOldify
ffmpeg
CUDA GPU support
Pexels API key
```

Some paths were written for Google Colab or Google Drive, especially in the first and fourth exercise notebooks. If I run them locally, I need to adjust those paths first.

