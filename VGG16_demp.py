from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import os

'''img = image.load_img("D:\\nn.jpg",target_size=(224,224))
img = np.asarray(img)
img = np.expand_dims(img,axis=0)
saved_model = load_model("D:\\OpenCV\\vgg.h5")
output = saved_model.predict(img)
val = sum(output[0])'''

cap = cv2.VideoCapture("E:/ball3.mp4")
no_of_frame = int(cap.get(7))

temp = []
    
i = 0
while i < no_of_frame:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(224,224))
    img = np.asarray(frame)
    img = np.expand_dims(img,axis=0)
    saved_model = load_model("E:\\OpenCV\\vgg.h5")
    output = saved_model.predict(img)
    val = sum(output[0])
    if val < 0.95:
        temp.append('Non-Pitch')
    else:
        temp.append('Pitch')
    print(str(round((i)/(no_of_frame)*100,2))+'% completed!')
    i+=1

cap1 = cv2.VideoCapture("E:/ball3.mp4")
no_of_frame1 = int(cap.get(7))    

i = 0
while i < no_of_frame1:
    ret,frame = cap1.read()
    frame = cv2.resize(frame,(720,480))
    value = temp[i]
    frame = cv2.putText(frame, value, (25, 25), cv2.FONT_ITALIC, 1, (0, 125, 255), 3,cv2.LINE_AA)
    cv2.imwrite('E:/pnp/'+str(i)+'.png',frame)
    cv2.imshow('Video',frame)
    key = cv2.waitKey(25)
    if key == ord('q'):
        break
    i+=1

cap.release()

cv2.destroyAllWindows()
