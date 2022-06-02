import numpy as np
import cv2
import os

path = "D:/xyz/"
mylist = os.listdir(path)
#print(mylist)

images = []
classes = []

for obj in mylist:
    img_path = path+obj
    cur_img = cv2.imread(img_path,0)
    images.append(cur_img)
    if obj[0] == 'n':
        classes.append('Non Pitch View')
    else:
        classes.append('Pitch View')


orb = cv2.ORB_create(nfeatures = 1000)

def find_des(images):
    deslist = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        deslist.append(des)
    return deslist

def find_ID(image,desList):
    kpimg,desimg = orb.detectAndCompute(image,None)
    bf = cv2.BFMatcher()
    match_list = []
    for des in desList:
        matches = bf.knnMatch(desimg, des, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        match_list.append(len(good))
    index = match_list.index(max(match_list))
    return classes[index]


'''img = cv2.imread("D:\\test_data\\pic1569.jpg",1)
img = img[0:415,0:715]
match = find_ID(img, des_list)
index = match.index(max(match))
print(classes[index])
#cv2.imshow('Test Image',img)
#cv2.waitKey() 
#cv2.destroyAllWindows()'''

des_list = find_des(images)
cap = cv2.VideoCapture("D:/vid1.mp4")
no_of_frame = int(cap.get(7))

op = []

i = 0
while i<no_of_frame:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(720,480))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img = gray[0:415,0:715]
    value = find_ID(img, des_list)
    frame = cv2.putText(frame, value, (25, 25), cv2.FONT_ITALIC, 1, (0, 125, 255), 3,cv2.LINE_AA)
    op.append(frame)
    print(str(round((i)/(no_of_frame)*100,2))+'% completed!')
    i+=1

print('100% completed!')

for frame in op:
    cv2.imshow('Video',frame)
    key = cv2.waitKey(25)
    if key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()











'''
img1 = cv2.imread("D:\\test_data\\pic476.jpg",0)
img2 = cv2.imread("D:\\test_data\\pic473.jpg",0)

img1 = img1[0:415,0:715]
img2 = img2[0:415,0:715]

orb = cv2.ORB_create(nfeatures = 1000)
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None)


imgkp1 = cv2.drawKeypoints(img1, kp1, None)
imgkp2 = cv2.drawKeypoints(img2, kp2, None)

#cv2.imshow('Pitch',imgkp1)
#cv2.imshow('Non Pitch',imgkp2)
cv2.imshow('map',img3)
cv2.waitKey() 
cv2.destroyAllWindows()'''