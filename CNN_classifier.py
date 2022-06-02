from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dir_path = "D:\\xyz"

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(
    "D:\\train_data",
    target_size=(480,480),
    batch_size=3,
    class_mode='binary')

validation_dataset = validation.flow_from_directory(
    "D:\\validation_data",
    target_size=(480,480),
    batch_size=3,
    class_mode='binary')

model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(480,480,3)),
     tf.keras.layers.MaxPool2D(2,2),
     #
     tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
     tf.keras.layers.MaxPool2D(2,2),
     #
     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
     tf.keras.layers.MaxPool2D(2,2),
     #
     tf.keras.layers.Flatten(),
     #
     tf.keras.layers.Dense(512,activation='relu'),
     #
     tf.keras.layers.Dense(1,activation='sigmoid'),
     ])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch=5,
                      epochs=30,
                      validation_data=validation_dataset)

'''count = 0
temp = []
for i in os.listdir(dir_path):
    count += 1
    temp.append(i)'''

'''#i = 0
for i in os.listdir(dir_path):
    img = cv2.imread(dir_path+'\\'+i,1)
    img = cv2.resize(img,(480,480))
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
        value = 'Non-Pitch'
    else:
        value = 'Pitch'
    img = cv2.putText(img, value, (25, 25), cv2.FONT_ITALIC, 1, (0, 125, 255), 3,cv2.LINE_AA)
    cv2.imshow('Video',img)
    key = cv2.waitKey(1000)
    if key == ord('q'):
        break
    #i+=1'''

cap = cv2.VideoCapture("D:/vid1.mp4")
no_of_frame = int(cap.get(7))
    
i = 0
while i < no_of_frame:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(480,480))
    X = image.img_to_array(frame)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
        value = 'Non-Pitch'
    else:
        value = 'Pitch'
    img = cv2.putText(frame, value, (25, 25), cv2.FONT_ITALIC, 1, (0, 125, 255), 3,cv2.LINE_AA)
    cv2.imshow('Video',img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    i+=1

'''for i in os.listdir(dir_path):
    img = cv2.imread(dir_path+'\\'+i,1)
    img = cv2.resize(img,(720,720)) 
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
        print('Non Pitch')
    else:
        print('Pitch')'''

#cv2.imshow("Original",img1)

cv2.waitKey() 
cv2.destroyAllWindows()