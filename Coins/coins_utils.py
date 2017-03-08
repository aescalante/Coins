from PIL import Image
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import copy
import time
import re
import random
import os
from keras.models import Model, model_from_json
from keras.optimizers import RMSprop
import glob

#Reading input data
def read_image(file_path, ROWS = 100, COLS = 100):
    img = cv2.imread(file_path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Needed to obtain right ordering in colors between opencv and matplotlib
    #Do we want to resize?
    #return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    return img

#Prepare the images for further processing
def prep_data(images, ROWS = 100, COLS = 100, CHANNELS = 3):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        if "jpg" in image_file:
            image = read_image(image_file)
            #print image.shape
            data[i] = image
            if i%1000 == 0: print('Processed {} of {}'.format(i, count))
        else:
            print(image_file + " Not processed")
    
    return data

#Print some coins (to be extended...) for peace of mind while labeling
def printSomeCoints(img, train_images_name, labels, ncoins):
    for i in range(0,ncoins):
        print(train_images_name[i])
        if labels[i] == 0.:
            print("This is 1 euro or 2 euro")
        if labels[i] == 1.:
            print("This is 50, 20 or 10 cent")
        if labels[i] == 2.:
            print("This is 5, 2, 1 cent")
        plt.figure(figsize=(10,5))
        print(train[i].shape)
        plt.imshow(train[i])
        plt.show()

#Save a cNN model        
def SaveModel(model, modelName):
    # serialize model to JSON
    model_json = model.to_json()
    with open(modelName+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelName+".h5",overwrite=True)
    print("Saved model "+modelName+" to disk")  
    return 0

#Load cNN model from a file
def LoadModel(modelName):
    # load json and create model
    json_file = open(modelName+".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(modelName+".h5")
    print("Loaded model "+modelName+" from disk \n")
    return loaded_model

#Generator for computing times
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference


TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def showInformation(img):
    print(img.shape) #dimensions of input image
    plt.imshow(img)
    plt.show()
    
def inter_centre_distance(x1,y1,x2,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def colliding_circles(circles):
    if circles is not None:
        for i in circles[0,:]:
            for j in circles[0,:]:
                x1, y1, Radius1 = i
                x2, y2, Radius2 = j
                if i[0] == j[0] and i[1] == j[1]:
                    continue ## looking at the same circle
                
                #collision or containment:
                #print inter_centre_distance(x1,y1,x2,y2), Radius1, Radius2
                #if inter_centre_distance(x1,y1,x2,y2) < (Radius1 + Radius2)*0.95:
                if inter_centre_distance(x1,y1,x2,y2) < (Radius1 + Radius2)*0.85:
                    print("Warning: There are colliding circles "+str(len(circles[0,:])))
                    return True
    else:
        print("Warning: No circles found!")


def drawCircles(img,circles):
    cimg=copy.deepcopy(img)
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,255,0),3)
    return cimg
            
def findCircles(img, param1=50, param2=30):
    minDist = int(img.shape[1]/15) # To test
    minRadius = int(img.shape[1]/35) # To test
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.dilate(gray, None, 1)
    gray = cv2.erode(gray, None, 1) 
    
    plt.imshow(gray, cmap='gray')
    plt.show()

    try:
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,0.01,minDist, param1=param1, param2=param2, minRadius=minRadius)
        showInformation(drawCircles(img, circles))
        nRefCircles = min(len(circles[0,:]),5)
        refRadius = np.mean(circles[0,0:(nRefCircles+1),2])
        minRadius = int(refRadius*0.8)
        maxRadius = int(refRadius*1.6)
        #maxRadius = int(min(circles[0,0:-1,2])*2) # 50% more than the smallest circle
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,0.01,minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    except:
        param2=int(param2/2) # largely reduce par2 to ensure convervence
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,0.01,minDist, param1=param1, param2=param2, minRadius=minRadius)    
        nRefCircles = min(len(circles[0,:]),5)
        refRadius = np.mean(circles[0,0:(nRefCircles+1),2])
        minRadius = int(refRadius*0.8)
        maxRadius = int(refRadius*1.6)
        #maxRadius = int(min(circles[0,0:-1,2])*2)
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,0.01,minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    #Suspect that I miss circles
    while len(circles[0,:]) in range(0,11) and param2>=70: 
        param2 = param2 -5
        #twice the separation to reduce fakes
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,0.01,minDist, param1=param1, param2=param2, minRadius=minRadius)    
    #Suspect that I have too many circles
    while colliding_circles(circles) == True:
        step = 0.5
        if len(circles[0,:]) > 30:
            step = 1
        if len(circles[0,:]) > 45:
            step = 2.5
        if len(circles[0,:]) > 60:
            step = 5
        param2 = param2 +step
        print("Trying para2... "+str(param2))
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,0.01,minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    ncircles = len(circles[0,:])
    #Decrease the criteria to built a circle (at the risk of getting fakes) when low number of circles found.

    circles = np.uint16(np.around(circles))
    print("Total number of circles "+str(len(circles[0,:])))
    return circles

def showCandidates(img,circles):
    candidates = []
    for i in circles[0,:]:
        print(i) 
        xcenter = i[1]
        ycenter = i[0]
        radius = i[2] 
        radius_ext = int(radius*1.30) # extend the radius by 15% to get border of coin
        xmin = max(0, xcenter-radius_ext)
        xmax = min(img.shape[0], xcenter+radius_ext)
        ymin = max(0, ycenter-radius_ext)
        ymax = min(img.shape[1], ycenter+radius_ext)
        print(xmin, xmax, ymin, ymax)
        plt.imshow(img[xmin:xmax, ymin:ymax])
        plt.show()
        candidates.append(img[xmin:xmax, ymin:ymax])
    return candidates

def saveCandidates(outputfolder, source, candidates):
    print("Saving Coins")
    sourceSplit = re.split(r'/|\\',source)
    #sourceSplit = source.split("/")
    #if sourceSplit == source:
    #    sourceSplit = source.split("\\")
    output_filename = outputfolder+sourceSplit[-1].split(".")[0]+"_Candidate_"
    print(output_filename)
    print(source)
    for index, i in enumerate(candidates):
        #print index, i.shape
        img = Image.fromarray(i, 'RGB')
        img.save(output_filename+str(index+1)+".jpg")
    return 0

def saveCircles(outputfolder, source, cimg):
    print("Saving Circles")
    sourceSplit = re.split(r'/|\\',source)
    #sourceSplit = source.split("/|\\")
    #if sourceSplit == source:
    #    sourceSplit = source.split("\\")
    output_filename = outputfolder+sourceSplit[-1].split(".")[0]+"_Circles"
    plt.imshow(cimg)
    img = Image.fromarray(cimg, 'RGB')
    img.save(output_filename+".jpg")
    return 0