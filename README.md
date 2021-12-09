# Video-content-classification-using-CNN-LSTM
### Classify input video with the according to its content type.

[![N|Solid](https://www.gstatic.com/devrel-devsite/prod/v0009e14c1212eb34a833a614ba55cbefddb8efdabe01fcac037dbc181c8c3153/tensorflow/images/lockup.svg)](https://nodesource.com/products/nsolid)




## Required Libraries


```sh
pip install tensorflow opencv-contrib-python pafy youtube-dl moviepy pydot
```



## Step 1: Visualize the Data with its Labels

we will visualize the data along with labels to get an idea about what we will be dealing with.

```sh
all_classes_names = os.listdir("file path to the Database")
```
Convert Video to frames
```sh
video_reader = cv2.VideoCapture(f'dataset/UCF50/{selected_class_Name}/{selected_video_file_name}')
```
Display the frames
 ```sh
plt.subplot(5, 4, counter);plt.imshow(rgb_frame);plt.axis('off')
```

## Step 2: Preprocess the Dataset
