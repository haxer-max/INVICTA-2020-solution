import cv2
import numpy as np

#These are thresholds for maple image which were manually calibrated
lowerMaple=np.array([33,143,73])
upperMaple=np.array([86,255,255])

#These are thresholds for maple image which were manually calibrated
lowerNeem=np.array([37,106,113])
upperNeem=np.array([50,255,255])

#the al=bouve thersholds will be used for feature 1


#these are ranges for non white colour in gray scale image
#these will be used to create mask to draw contours
l=np.array([0])
u=np.array([240])







#this is the function that does all the computation for 1 set.
# we have 2 sets, so we will call it twice 
def DoEveryThing(img,img_org,lowerBound,upperBound,outname):
    #
    #COMPUTATIONS ON PERFECT IMAGE
    imgGRAY_org=cv2.cvtColor(img_org,cv2.COLOR_BGR2GRAY)     #convert to grayscale
    mask_org = cv2.inRange(imgGRAY_org, l,u)                  #mask to identify where leaves are present
    contours_org, hierarchy = cv2.findContours(mask_org, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   #generating contours on with help of image
                                                                                                     #where we identified where leaves are present
    imgcpy_org=img_org.copy()#copying image so as to not harm the oriignal one
    
    #perimeter and area of perfect image
    area_org=0
    perimeter_org=0
    #below part is almost asme as level1
    #looping is done to discard false contour, note that we only have 1 true contour here 
    for cnt in contours_org:
        area=cv2.contourArea(cnt)        
        if(area>50):
            cv2.drawContours(imgcpy_org,cnt,-1,(255,0,0),3)
            #compute area and perimeter
            area_org=area
            perimeter_org=cv2.arcLength(cnt,True)
    Ap_perfect=area/perimeter_org**2 #this will be our feature 2
    
    
    #
    #COMPUTATIONS ON GIVEN SET OF IMAGES
    #the lots of below fuctions were almost same as task 1 
    imgGRAY=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgGRAY, l,u)                      #mask to detect leaves
    maskg = cv2.inRange(imgHSV, lowerBound,upperBound)   #generate mask where colors are in threshold
                                                         #too dark or too ligh colors will be discarded 
                                                         
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #generating contour on mask where leaf is detected
    imgcpy=img.copy()
    
    for cnt in contours:
        area=cv2.contourArea(cnt)   
             
        if(area>200):   #this is just to make sure we dont compute on false contours
            x,y,w,h=cv2.boundingRect(cnt) #gitting coordinates of rectangle
            cv2.rectangle(imgcpy,(x,y),(x+w,y+h),(0,0,255)) #drawing rectangle
            G=np.sum(maskg[y:y+h+1,x:x+w+1])            #this will check for the threshold match. too dark or too light will score low
            T=np.sum(mask[y:y+h+1,x:x+w+1])             #this is total area of leaf
            score1=G/T                                  #score for feature one
            
            #calc perimeter and area for feature 2
            perimeter=cv2.arcLength(cnt,True)
            Ap=area/perimeter**2                           #feature2

            #score for feature 2
            score2=abs(Ap-Ap_perfect)/Ap_perfect
            score2=1-score2     

            score=100*score1*score2*2/(score1+score2)   #using harmonic mean for combining the score (analogus to F1 score)
            cv2.putText(imgcpy, str(round(score,1))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0), 2)   #writing scores
    cv2.imwrite(outname,imgcpy)         #outputting the final image
    







#MAPLE
#importing the image
imgmaple =cv2.imread("images/mapleleafcorrect.jpg")
imgB =cv2.imread("images/mapleleaves.jpg")
#calling the function
DoEveryThing(imgB,imgmaple,lowerMaple,upperMaple,"output/Mapleout.jpg")

#NEEM
#importing the image
imgneem =cv2.imread("images/neemleafcorrect.jpg")
imgC =cv2.imread("images/neemleaves.jpg")
#calling the function
DoEveryThing(imgC,imgneem,lowerNeem,upperNeem,"output/Neemout.jpg")



