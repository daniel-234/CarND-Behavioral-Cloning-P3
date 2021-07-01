import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Conv2D
import csv
import cv2
    
# Store the rows of the csv file in a list. 
lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

for line in lines:
    # Read the path to the center image. 
    filename = line[0]
    # Build the local path to the image.  
    local_path = './data/' + filename
    # Read the image and append ot to the list. 
    image = cv2.imread(local_path)
    # Convert the image to RGB format, as the file that handles
    # simulation uses that format (while cv2 uses BGR). 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image_rgb)
    # Read the 4th column from the CSV file, that is
    # the stearing angle and save it to another list. 
    measurement = line[3]
    measurements.append(measurement)
    # Augment the dataset for a more robust set of measurements.
    # Flip the image.
    image_flipped = cv2.flip(image_rgb, -1)
    # Take the opposite sign of the steering measurement. 
    measurement_flipped = float(measurement) * -1.0
    # Append the results to the images and measurements lists. 
    images.append(image_flipped)
    measurements.append(measurement_flipped)
    
# Create the training and label Numpy arrays. 
X_train = np.array(images)
y_train = np.array(measurements)

# Check that the X_train array has double the elements of lines.
print(len(lines))
print(X_train.shape)

#print(X_train.shape)
model = Sequential()
# Add a Lambda layer and parallelize image normalization during training. 
# pixel_normalized = pixel / 255
# Center the image at 0 with pixels in range [-0.5, 0.5]
# pixel_mean_centered = pixel_normalized - 0.5
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320,3)))

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