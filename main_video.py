#import sys
#sys.path.append('/home/mohilbajaj2002/anaconda3/envs/computervision/lib/python2.7/site-packages')
#sys.path.append('/home/mohilbajaj2002/anaconda3/pkgs')


import numpy as np
import cv2
import pickle

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


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=7)
    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start:ycord_end, xcord_start:xcord_end)
    	roi_color = frame[y:y+h, x:x+w]

    	color = (255, 0, 0) #BGR 0-255 
	font = cv2.FONT_HERSHEY_SIMPLEX
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

	for i in range(0,len(features)):
		subitem = features[i].detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=7)
    		for (ex,ey,ew,eh) in subitem:
    			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			cv2.putText(roi_color, f_name[i], (ex,ey), font, 0.3, (255,255,255), stroke, cv2.LINE_AA)

	subitem = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=18)
    	for (ex,ey,ew,eh) in subitem:
    			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			cv2.putText(roi_color, 'nose', (ex,ey), font, 0.3, (255,255,255), stroke, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Display Box',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




