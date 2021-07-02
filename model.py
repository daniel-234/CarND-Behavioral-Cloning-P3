import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Conv2D, Cropping2D
import csv
import cv2
    
# Store the images and steering measurements of the csv file in lists. 
images = []
measurements = []

# Read an image and its measured steering angle.    
def process_image(image_name, steering_angle):
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
    image_center, image_left, image_right = line[0].strip(), line[1].strip(), line[2].strip()
    process_image(image_center, steering_center)
    process_image(image_left, steering_left)
    process_image(image_right, steering_right)

# Open the CSV file.
with open('./data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        process_line(line)
    
# Create the training and label Numpy arrays. 
X_train = np.array(images)
y_train = np.array(measurements)

# Check that the X_train array has double the elements of lines.
print(len(images))
print(X_train.shape)

#print(X_train.shape)
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

model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten the input.
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
# As we're doing regression here, we only need 1 output. 
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')