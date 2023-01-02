import cv2 
from glob import glob
import numpy as np
import pickle as pkl
path='F:\\projects\\NEW_UNET\\stage1_train\\'
x=[]
y=[]
for i in glob(path+"*"):
    temp=np.zeros((128,128))
    for j in glob(i+"\masks\*"):
        img=cv2.imread(j,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(128,128))
        temp=temp+img
    y.append(temp)
    img=cv2.imread(glob(i+"\images\*")[0])
    img=cv2.resize(img,(128,128))
    x.append(img)


pkl.dump(x,open("X_128.dat","wb"))
pkl.dump(y,open("Y_128.dat","wb"))