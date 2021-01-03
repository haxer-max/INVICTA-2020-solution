import cv2
import numpy as np

#These are ranges for green color manually calibrated
lowerBound1=np.array([36,55,33])
upperBound1=np.array([90,255,255])

#These are ranges for yellow color manually calibrated
lowerBound2=np.array([10,100,0])
upperBound2=np.array([35,255,255])

#these are ranges for non white colour in gray scale image
#these will be used to create mask to draw contours
l=np.array([0])
u=np.array([230])



#this is the function that does everthing 
def DoEveryThing(img):
    imgGRAY=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                #convert to grayscale
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)                  #convert to hsv format
    mask = cv2.inRange(imgGRAY, l,u)                            #mask to identify where leaves are present
    maskg = cv2.inRange(imgHSV, lowerBound1,upperBound1)        #mask to identify where green is present
    masky = cv2.inRange(imgHSV, lowerBound2,upperBound2)        #mask to identify where yellow is present

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #generating contours on with help of image
                                                                                            #where we identified where leaves are present
    imgcpy=img.copy()   #copying image so as to not harm the oriignal one
    
    for cnt in contours:        #looping in list of contours
        area=cv2.contourArea(cnt)        #for each contour, we calculate area 
        
        if(area>50):                    #this is just to make sure we dont compute on false contours
            cv2.drawContours(imgcpy,cnt,-1,(255,0,0),3)             #drawing the contours on the copied image
            x,y,w,h=cv2.boundingRect(cnt)                           #geting coordinates of rectangle
            cv2.rectangle(imgcpy,(x,y),(x+w,y+h),(0,0,255))         #drawing rectangle
            G=np.sum(maskg[y:y+h+1,x:x+w+1])                        #finding amount of green pixels from the mask for green in that rectangle
            Y=np.sum(masky[y:y+h+1,x:x+w+1])                        #finding amount of yellow pixels from the mask for yellow in that rectangle 
            score=G/(G+Y)                                           #formula given in problem statement 
            
            name="Old: "
            if score>0.7:                                           #checking if leaf is new or old
                name="Fresh: "
            name+=(str(round(score,2)))                             
            cv2.putText(imgcpy, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0), 2)     #this is for labeling the image with scores
    
    cv2.imshow("contored",imgcpy)                                                           #displaying the image 
    cv2.imwrite("output/contored.jpg",imgcpy)                                                       #saving the image   
    cv2.waitKey(0)


#importing the image
img =cv2.imread("images/level1.jpg")
#calling the function
DoEveryThing(img)
