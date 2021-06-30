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
    img_path = line[0]
    # Split the path and get the tokens.  
    tokens = img_path.split('/')
    # Get the last token (the filename).
    filename = tokens[-1]
    # Build the local path to the image.  
    local_path = './data/' + filename
    # Read the image and append ot to the list. 
    image = cv2.imread(local_path)
    images.append(image)
    # Read the 4th column from the CSV file, that is
    # the stearing angle and save it to another list. 
    measurement = line[3]
    measurements.append(measurement)
    
print(len(images))
print(len(measurements))
    