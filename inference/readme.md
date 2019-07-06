## Introduce
This folder is a demo for use ssd-vgg as a image detection.

## How to use
### Image
```buildoutcfg
python inference.py -i ./images/eagle.jpg
```
The train dataset is based on Pascal VOC datasets
the catogory num and name please refer to this type.py
### Video
if you want to input a new video
```buildoutcfg
python inference.py -i ./videos/*.mp4 -t 2
```
if you want to process the video and save to a new video:
```buildoutcfg
python inference.py -i ./videos/*.mp4 -t 3
```
you can find the output_*.mp4 in the videos directory

### Read the camera data and detection
```buildoutcfg
python inference.py -t 4
```