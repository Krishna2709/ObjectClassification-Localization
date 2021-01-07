# 1. Prepare a drawing function
# 2. Load the model and connect it to the camera
# 3. Loop over the frames from the video stream
# 4. For each frame of the stream, the model make an inference and
# drawing function to display the results.

# a. Import required modules
import numpy as np
import cv2
import tensorflow.keras as K

# b. Function to draw localization bounding boxes on a frame
def draw_box(frame:np.ndarray, box:np.ndarray) -> np.ndarray:
  '''
  Input: frame and normalized coordinates of the two corners of a bounding box,
          as an array of four numbers.
  Process: 1. reshapes the 1D array of the box into 2D array
              (first index represents the point and the second represents the x and y coordinates) 
           2. transforms the normalized coordinates to coordinates of image by multiplying with w, h
           3. draws the green coloured bounding box
  Output: frame with bounding box
  '''
  h,w = frame.shape[0:2]
  pts = (box.reshape((2,2))*np.array([w,h])).astype(np.int)
  cv2.rectangle(frame, tuple(pts[0]), tuple(pts[1]), (0,255,0), 2)
  return frame

# c. Import the model and connect to camera
model = K.models.load_model("localization.h5")

cap = cv2.VideoCapture(0)

# iterate over the frames from the camera,
for _, frame in iter(cap.read, (False, None)):
  # resize each frame to a standard size
  input = cv2.resize(frame, (224,224))
  # convert frame to RGB color space
  input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

  # normalize the image and add another dimension since the model accepts batches of images,
  # and pass the result to the model for inference
  box, = model.predict(input[None]/255)
  # drawing the predicted box
  draw_box(frame, box)
  # display
  cv2.imshow("res", frame)
  if(cv2.waitKey(1) == 27):
    break