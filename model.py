import numpy as np
import math
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Conv2D, Cropping2D, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
import cv2
import matplotlib.pyplot as plt
    
samples = []

# Read an image and its measured steering angle.    
def process_image(image_name, steering_angle):
    images = [] 
    measurements = []
    
    # Build the local path to the image.  
    local_path = './data/' + image_name
    # Read the image and append ot to the list. 
    image = cv2.imread(local_path)
    # Convert the image to RGB format, as the file that handles
    # simulation uses that format (while cv2 uses BGR). 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image_rgb)    
    measurements.append(steering_angle)
    # Augment the dataset for a more robust set of measurements.
    # Flip the image.
    image_flipped = cv2.flip(image_rgb, -1)
    # Take the opposite sign of the steering measurement. 
    steering_angle_flipped = steering_angle * -1.0
    # Append the results to the images and measurements lists. 
    images.append(image_flipped)
    measurements.append(steering_angle_flipped)
        
    return images, measurements


# Read images and measurements from a row of the CSV file. 
def process_line(line):  
    # Read the 4th column from the CSV file, that is the stearing 
    # center angle and save it to the measurements list. 
    steering_center = float(line[3])
    # Create a correction coefficient to apply to the left and right cameras.
    # The purpose of this correction is to teach the model to steer and recover
    # when it gets a little bit off track from the center.
    steering_correction = 0.06
    # Apply a positive correction to the left image to make it go back to the
    # center towards the right.
    steering_left = steering_center + steering_correction
    # Apply a negative correction to the right image to make it go towards the left.
    steering_right = steering_center - steering_correction
    # Read images from the center, left and right cameras (N.B. left and right paths 
    # have leading space that needs to be removed).
    image_center = line[0].strip()
    image_left = line[1].strip()
    image_right = line[2].strip()
    
    images_center, measurements_center = process_image(image_center, steering_center)
    images_left, measurements_left = process_image(image_left, steering_left)
    images_right, measurements_right =  process_image(image_right, steering_right)
    
    return images_center, images_left, images_right, measurements_center, measurements_left, measurements_right

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1: # The generator never terminates
        # Shuffle the whole dataset.
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            
            # Store the images and steering measurements of the csv file in lists. 
            images = []
            measurements = []
            
            for batch_sample in batch_samples:                
                images_center, images_left, images_right, measurements_center, measurements_left, measurements_right = process_line(batch_sample)
                
                # Extend the images and measurements array with the new images collected (instead of appending them as new arrays).
                # See https://stackoverflow.com/questions/252703/what-is-the-difference-between-pythons-list-methods-append-and-extend
                images.extend(images_center)
                images.extend(images_left)
                images.extend(images_right)
                measurements.extend(measurements_center)
                measurements.extend(measurements_left)
                measurements.extend(measurements_right)
                
            # Create the training samples and labels Numpy arrays. 
            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield shuffle(X_train, y_train)
    
# Open the CSV file.
with open('./data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
        
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
        
# Set the desired batch size.
batch_size = 64
# Create generators for the training and validation data.
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)

# Create a L2 reguralizer
reg = l2(0.01)

model = Sequential()
# Crop the images, as the top portion captures trees, hills and the sky,
# while the bottom portion captures the hood of the car. 
# Set up cropping layer. By adding the functionality to crop the images
# to the model, we can use the parallelization offered by the GPU so 
# that many images are cropped simultaneously. 
model.add(Cropping2D(cropping=((50, 20), (0,0)), input_shape=(160, 320, 3)))
# Add a Lambda layer and parallelize image normalization during training. 
# pixel_normalized = pixel / 255
# Center the image at 0 with pixels in range [-0.5, 0.5]
# pixel_mean_centered = pixel_normalized - 0.5
# After cropping vertically, images have now shape (90, 320, 3)
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(90, 320, 3)))
# NVidia training architecture. 
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(64, (5, 5), activation='relu', activity_regularizer=l2(0.01)))
# Flatten the input.
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
# As we're doing regression here, we only need 1 output. 
model.add(Dense(1))

opt = SGD(lr = 0.001, momentum = 0.9)
model.compile(optimizer=opt, loss='mean_squared_logarithmic_error')

model.summary()

callback = EarlyStopping(monitor='val_loss', patience = 2)

history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), 
                    epochs=10, verbose=1, callbacks=[callback])

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')