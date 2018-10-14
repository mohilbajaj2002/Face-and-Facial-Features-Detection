import sys
sys.path.append('/home/mohilbajaj2002/anaconda3/envs/computervision/lib/python2.7/site-packages')
sys.path.append('/home/mohilbajaj2002/anaconda3/pkgs')

import numpy as np
import cv2

# Get user values for image path

path = sys.argv[1]

# Importing Cascades
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
left_eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_righteye_2splits.xml')
smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')
nose_cascade = cv2.CascadeClassifier('cascades/Nose18x15.xml')
left_ear_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_rightear.xml')

features = [left_eye_cascade, right_eye_cascade, smile_cascade, left_ear_cascade, right_ear_cascade]
f_name = ['left_eye', 'right_eye', 'smile', 'left_ear', 'right_ear']


image = cv2.imread(path)

gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=3)
for (x, y, w, h) in faces:
	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start:ycord_end, xcord_start:xcord_end)
    	roi_color = image[y:y+h, x:x+w]

    	color = (255, 0, 0) #BGR 0-255 
	font = cv2.FONT_HERSHEY_SIMPLEX
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(image, (x, y), (end_cord_x, end_cord_y), color, stroke)

	for i in range(0,len(features)):
		subitem = features[i].detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5)
    		for (ex,ey,ew,eh) in subitem:
    			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			cv2.putText(roi_color, f_name[i], (ex,ey), font, 0.3, (255,255,255), stroke, cv2.LINE_AA)

	subitem = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
    	for (ex,ey,ew,eh) in subitem:
    			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			cv2.putText(roi_color, 'nose', (ex,ey), font, 0.3, (255,255,255), stroke, cv2.LINE_AA)

# Display the resulting image
cv2.imshow('Image',image)
cv2.waitKey(10000) 

# When everything done, release the windows
cv2.destroyAllWindows()




