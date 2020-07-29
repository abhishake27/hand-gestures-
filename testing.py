import cv2
from keras.models import load_model
import numpy as np

cam = cv2.VideoCapture(0)

test_model = load_model('F:/spycodes/MyModel.h5')
start = False
while True:
    ret, test_image = cam.read();
    
    cv2.rectangle(test_image,(47,75),(277,305),(255,255,255),2)
    if start :
        roi = test_image[78:302,50:274]
        
        img_arr = np.array(roi)
        img_arr = np.expand_dims(img_arr,axis =0)
    
        pred = test_model.predict(img_arr)
        #print(pred)
        if(pred[0][0]>0.5):
            cv2.putText(test_image,'paper', (30,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,60),4)
        elif (pred[0][1]>0.5):
            cv2.putText(test_image,'rock', (30,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,60),4)
        elif (pred[0][2]>0.5):
            cv2.putText(test_image,'scissors', (30,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,60),4)
        
    cv2.imshow('test', test_image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):
        start = True
        
    
cam.release()
cv2.destroyAllWindows()
