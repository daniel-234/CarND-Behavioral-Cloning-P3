# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Neural_Network.png "Model Visualization"
[image2]: ./examples/center_lane_driving.jpg "Center lane driving"
[image3]: ./examples/recovering_to_the_center_1.jpg "Recovery Image"
[image4]: ./examples/recovering_to_the_center_2.jpg "Recovery Image"
[image5]: ./examples/recovering_to_the_center_3.jpg "Recovery Image"
[image6]: ./examples/recovering_to_the_center_4.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **`model.py`**  containing the script to create and train the model
* **`drive.py`** for driving the car in autonomous mode
* **`model.h5`** containing a trained convolution neural network 
* **`writeup_report.md`** or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my **`drive.py`** file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The **`model.py`** file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the same [end-to-end deep learning network for self-driving cars developed by NVidia](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). 

The model consists of 9 layers, including a normalization layer, 5 convolutional layers with 3x3 and 5x5 filter sizes and depths between 24 and 64 and 3 fully connected layers (**`model.py`** lines 114-143). 

The model includes RELU layers to introduce nonlinearity (code lines 128-140), and the data is normalized in the model using a Keras lambda layer (code line 126). 

#### 2. Attempts to reduce overfitting in the model

The model contains Weight and Noise regularization in order to reduce overfitting (**`model.py`** lines 128-141). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 151-153). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (**`model.py`** line 145).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and one lap focusing on driving smoothly around curves. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a small neural network that would at least drive the car a little.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.  
Unfortunately the car was only able to drive for a couple of meters until it steered away.

My next step was to normalize the data and use the LeNet architecture to see if things improved. 
With LeNet the car was able to run for some more meters before steering away. 

I then added preprocessing to the data, by augmenting it with images and measurements flipping, cropping the images to avoid parts that weren't useful for training (the car hood at the bottom and trees and the sky at the top) and the collection of data from the left and right cameras, as well. 
The car performed better, but its behavior didn'look well at the first curve. 

So I tried a more powerful network architecture, [published](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) by the Autonomous Vehicle Team at NVidia. 

I found that the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it could generalize better. 
At first I added Dropout, without having the results I was looking for. I then tried activity regularization and weight regularization and found that the solution that worked better was to use only kernel regularization with an L2 vector. 
Then I added Noise Regularization to gain better robustness. 

The final step was to run the simulator again to see how well the car was driving around track one. There were a few spots, precisely the steepest curves, where the vehicle fell off the track. 
To improve the driving behavior in these cases, I decided to collect a new set of data. Until now, in fact, I was using the data provided in the course. 
So I collected two or three laps of center lane driving, one lap of recovery driving from the sides and
one lap focusing on driving smoothly around curves, to improve the car ability on turning data. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 114-143) consisted of a convolution neural network with the following layers and layer sizes: 
- A Cropping Layer with input images 160x320x3 and output 65x320x3
- A Normalization Layer
- A Convolutional layer with 24 filters, a 2x2 stride and a 5x5 kernel
- A Convolutional layer with 36 filters, a 2x2 stride and a 5x5 kernel
- A Convolutional layer with 48 filters, a 2x2 stride and a 5x5 kernel
- Convolutional layer with 64 filters and a 3x3 kernel
- Convolutional layer with 64 filters and a 3x3 kernel
- A Flatten layer
- A Fully-connected layer with 100 neurons
- A Fully-connected layer with 50 neurons
- A Fully-connected layer with 10 neurons

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would help with generalizing the data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by empiric trials and the loss metric. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Credits

[End-to-end Deep Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)

[How to use weight decay to reduce overfitting of Neural Networks in Keras](https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/)

[BGR to RGB conversion](https://stackoverflow.com/questions/50963283/python-opencv-imshow-doesnt-need-convert-from-bgr-to-rgb)