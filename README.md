# Human Action recognition using ensemble nerural networks
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
 we will read the video files from the dataset and resize the frames of the videos to a fixed width and height, to reduce the computations and normalized the data to range
 
###  Function to Extract, Resize & Normalize Frames
 
```sh
def frames_extraction(video_path):
```
###  Function to Extract, Resize & Normalize Frames
```sh
create_dataset()
``` 

#### this will return return the frames (features), class index ( labels), and video file path (video_files_paths).

## Step 3: Split the Data into Train and Test Set
```sh
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                            test_size = 0.25, shuffle = True,
                                                                            random_state = seed_constant)
``` 
.




## Step 4: Implement the ConvLSTM Approach


### Step 4.1: Construct the Model

Use Keras ConvLSTM2D recurrent layers.

```sh
 create_convlstm_model()
 ```
#### Check Model’s Structure:
```sh
plot_model(convlstm_model, to_file = 'convlstm_model_structure_plot.png', show_shapes = True, show_layer_names = True)
 ```



### Step 4.2: Compile & Train the Model
```sh
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)

convlstm_model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

convlstm_model_training_history = convlstm_model.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4,
                                                     shuffle = True, validation_split = 0.2, 
                                                     callbacks = [early_stopping_callback])
```
#### Evaluate the Trained Mode

```sh
model_evaluation_history = convlstm_model.evaluate(features_test, labels_test)
 ```
#### Save the Model
```sh
model_file_name = f'convlstm_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'

# Save your Model.
convlstm_model.save(model_file_name)
 ```
### Step 4.3: Plot Model’s Loss & Accuracy Curves
plot_metric() to visualize the training and validation metrics.
```sh
plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name)
 ```
 
 #### Visualize the training and validation loss metrices.
 ```sh
plot_metric(convlstm_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
 ```
 
# Step 6: Test the Best Performing Model on random YouTube videos
#### download_youtube_videos() to download the YouTube videos first using pafy library
 ```sh
download_youtube_videos(youtube_video_url, output_directory):
 ```
 ## Testing system in under development , will be updatd soon 
## License

MIT

**Free Software, Hell Yeah!**


