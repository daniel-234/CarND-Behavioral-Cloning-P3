import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense
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
    images.append(image)
    # Read the 4th column from the CSV file, that is
    # the stearing angle and save it to another list. 
    measurement = line[3]
    measurements.append(measurement)
    
# Create the training and label Numpy arrays. 
X_train = np.array(images)
y_train = np.array(measurements)

#print(X_train.shape)
model = Sequential()
# Pass the X_train dimensions as input shape.
model.add(Flatten(input_shape=(160, 320,3)))
# As we're doing regression here, we only need 1 output. 
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')