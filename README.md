# Multi-branch Attention Network (MBAN) for small object detection

# Dataset

1. PASCAL VOC: http://host.robots.ox.ac.uk/pascal/VOC/
2. NWPU VHR-10: https://github.com/Gaoshuaikun/NWPU-VHR-10

# Code
The code is in the branch master.

# Train
1. This article uses the VOC format for training Before training, put the label file in the Annotation under the VOC2007 folder under the VOCdevkit folder. Before training, put the image files in JPEGImages under the VOC2007 folder under the VOCdevkit folder.
2. Use voc_annotation.py to get 2007_train.txt and 2007_val.txt for training.
3. start network training.
   
# Predict
1. In the mban.py file, modify model_path and classes_path to correspond to the trained files.
2. run predict.py.

# Evaluate
The evaluation results can be obtained by running get_map.py, and the evaluation results will be saved in the map_out folder.
