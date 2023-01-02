from keras.layers import Dense,Conv2D,Conv2DTranspose,Cropping2D,Input,LeakyReLU,BatchNormalization,MaxPool2D,Concatenate,Dropout
from keras.models import Model
import tensorflow as tf
import pickle as pkl
import cv2
import numpy as np
from sklearn.model_selection import train_test_split    
x=pkl.load(open("X_128.dat","rb"))
y=pkl.load(open("Y_128.dat","rb"))
x=np.array(x)
y=np.array(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

def get_indices(lay1,lay2):
    return (lay1.shape[1]-lay2.shape[1])/2

def dice_coefficient(y_true,y_pred):
    y1=y_true
    y2=y_pred
    return (2*tf.math.reduce_sum(y1*y2)+1)/(tf.math.reduce_sum(y1)+tf.math.reduce_sum(y2)+1)

def dice_loss(y_true,y_pred):
    return 1-dice_coefficient(y_true,y_pred)


inp=Input(shape=(128,128,3))

lay1=Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(inp)
lay1=Dropout(.3)(lay1)
lay1=BatchNormalization()(lay1)
lay1=Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay1)
lay1=Dropout(.3)(lay1)
lay1=BatchNormalization()(lay1)

lay2=MaxPool2D(pool_size=(2,2))(lay1)
lay2=Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay2)
lay2=Dropout(.3)(lay2)
lay2=BatchNormalization()(lay2)
lay2=Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay2)
lay2=Dropout(.3)(lay2)
lay2=BatchNormalization()(lay2)

lay3=MaxPool2D(pool_size=(2,2))(lay2)
lay3=Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay3)
lay3=Dropout(.3)(lay3)
lay3=BatchNormalization()(lay3)
lay3=Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay3)
lay3=Dropout(.3)(lay3)
lay3=BatchNormalization()(lay3)

lay4=MaxPool2D(pool_size=(2,2))(lay3)
lay4=Conv2D(512,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay4)
lay4=Dropout(.3)(lay4)
lay4=BatchNormalization()(lay4)
lay4=Conv2D(512,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay4)
lay4=Dropout(.3)(lay4)
lay4=BatchNormalization()(lay4)

lay5=Conv2DTranspose(256,(3,3),strides=(2,2),padding='same',activation='relu',kernel_initializer='he_normal')(lay4)
la5=Concatenate()([lay3,lay5])
lay5=Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay5)
lay5=Dropout(.3)(lay5)
lay5=BatchNormalization()(lay5)
lay5=Conv2D(256,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay5)
lay5=Dropout(.3)(lay5)
lay5=BatchNormalization()(lay5)

lay6=Conv2DTranspose(256,(3,3),strides=(2,2),padding='same',activation='relu',kernel_initializer='he_normal')(lay5)
la6=Concatenate()([lay2,lay6])
lay6=Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay6)
lay6=Dropout(.3)(lay6)
lay6=BatchNormalization()(lay6)
lay6=Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay6)
lay6=Dropout(.3)(lay6)
lay6=BatchNormalization()(lay6)

lay7=Conv2DTranspose(32,(3,3),strides=(2,2),padding='same',activation='relu',kernel_initializer='he_normal')(lay6)
la7=Concatenate()([lay1,lay7])
lay7=Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay7)
lay7=Dropout(.3)(lay7)
lay7=BatchNormalization()(lay7)
lay7=Conv2D(16,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(lay7)
lay7=Dropout(.3)(lay7)
lay7=BatchNormalization()(lay7)
op=Conv2D(1,(1,1),padding='same',activation='sigmoid')(lay7)

model=Model(inp,op)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')
model.fit(x_train,y_train/255,validation_data=(x_test,y_test/255),epochs=20,batch_size=32)



test=np.array(x[10])
test=np.reshape(test,(1,128,128,3))
p=model.predict(test)
cv2.imshow("predicted img",p[0,:,:,0])
cv2.imshow("true image",y[10])
cv2.imshow("actual  image",x[10])
if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()








