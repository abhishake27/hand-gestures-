import cv2
import os
import time
camera=cv2.VideoCapture(0)
i=1000
start = False
#for i in range(50):
while True :
    ret, img = camera.read()
    
    cv2.rectangle(img,(47,75),(277,305),(255,255,255),2)
    
    if not start:
       cv2.putText(img,"Press 's' to start collecting images"
                   ,(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(100,0,200),2) 
    
       
    if start :
       time.sleep(0.1)
       roi = img[78:302,50:274]
       cv2.putText(img,"collecting image:"+str(i),(0,50)
                   ,cv2.FONT_HERSHEY_SIMPLEX,2,(100,0,200),3)
       
       path = 'F:/rockpapersc/valid_data/scissors'
       #path = 'F:/rockpapersc/valid_data/rock'
       #path = 'F:/rockpapersc/valid_data/scissors' #my destination
       cv2.imwrite(os.path.join(path , str(i)+'.jpg'), roi)
       i=i+1
       
    cv2.imshow("cam",img)  
    
    #if i == 20:
     #   break

    k = cv2.waitKey(10)
    if  k == ord('s'):
       start = True
    elif k == ord('p'):
        start = False
        
    elif k == ord('q'):
       break

    
camera.release()
cv2.destroyAllWindows()