from keras.layers import Dense,Conv2D,Conv2DTranspose,Cropping2D,Input,LeakyReLU,BatchNormalization,MaxPool2D,Concatenate
from keras.models import Model
import tensorflow as tf
import pickle as pkl
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
x=pkl.load(open("X.dat","rb"))
y=pkl.load(open("Y.dat","rb"))
# y=pkl.load(open("Y_282.dat","rb"))
x=np.array(x)
y=np.array(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

def get_indices(lay1,lay2):
    return (lay1.shape[1]-lay2.shape[1])/2

def dice_coefficient(y_true,y_pred):
    # y1=tf.experimental.numpy.ravel(y_true)
    # y2=tf.experimental.numpy.ravel(y_pred)
    y1=y_true
    y2=y_pred
    return (2*tf.math.reduce_sum(y1*y2)+1)/(tf.math.reduce_sum(y1)+tf.math.reduce_sum(y2)+1)

def dice_loss(y_true,y_pred):
    return -dice_coefficient(y_true,y_pred)


inp=Input(shape=(320,320,3))

lay1=Conv2D(32,(3,3),padding="same")(inp)
lay1=LeakyReLU(alpha=.1)(lay1)
lay1=BatchNormalization()(lay1)
lay1=Conv2D(64,(5,5))(lay1)
lay1=LeakyReLU(alpha=.1)(lay1)
lay1=BatchNormalization()(lay1)

lay2=MaxPool2D(pool_size=(2,2))(lay1)
lay2=Conv2D(64,(3,3),padding="same")(lay2)
lay2=LeakyReLU(alpha=.1)(lay2)
lay2=BatchNormalization()(lay2)
lay2=Conv2D(128,(5,5))(lay2)
lay2=LeakyReLU(alpha=.1)(lay2)
lay2=BatchNormalization()(lay2)

lay3=MaxPool2D(pool_size=(2,2))(lay2)
lay3=Conv2D(128,(3,3),padding="same")(lay3)
lay3=LeakyReLU(alpha=.1)(lay3)
lay3=BatchNormalization()(lay3)
lay3=Conv2D(256,(6,6))(lay3)
lay3=LeakyReLU(alpha=.1)(lay3)
lay3=BatchNormalization()(lay3)

lay4=MaxPool2D(pool_size=(2,2))(lay3)
lay4=Conv2D(256,(3,3),padding="same")(lay4)
lay4=LeakyReLU(alpha=.3)(lay4)
lay4=BatchNormalization()(lay4)
lay4=Conv2D(512,(5,5))(lay4)
lay4=LeakyReLU(alpha=.1)(lay4)
lay4=BatchNormalization()(lay4)

lay5=Conv2DTranspose(128,(3,3),strides=(2,2),padding="same")(lay4)
crp=int(get_indices(lay3,lay5))
lay3crp=Cropping2D((crp,crp))(lay3)
lay5=Conv2D(128,(7,7),padding='same')(lay5)
lay5=LeakyReLU(alpha=.1)(lay5)
lay5=BatchNormalization()(lay5)
lay5=Conv2D(128,(3,3))(lay5)
lay5=LeakyReLU(alpha=.1)(lay5)
lay5=BatchNormalization()(lay5)

lay6=Conv2DTranspose(64,(3,3),strides=(2,2),padding="same")(lay5)
crp=int(get_indices(lay2,lay6))
lay2crp=Cropping2D((crp,crp))(lay2)
lay6=Conv2D(64,(7,7),padding='same')(lay6)
lay6=LeakyReLU(alpha=.1)(lay6)
lay6=BatchNormalization()(lay6)
lay6=Conv2D(64,(3,3))(lay6)
lay6=LeakyReLU(alpha=.1)(lay6)
lay6=BatchNormalization()(lay6)

lay7=Conv2DTranspose(64,(3,3),strides=(2,2),padding="same")(lay6)
crp=int(get_indices(lay1,lay7))
lay1crp=Cropping2D((crp,crp))(lay1)
lay7=Conv2D(32,(7,7),padding='same')(lay7)
lay7=LeakyReLU(alpha=.3)(lay7)
lay7=BatchNormalization()(lay7)
lay7=Conv2D(32,(3,3))(lay7)
lay7=LeakyReLU(alpha=.1)(lay7)
lay7=BatchNormalization()(lay7)

op=Conv2D(1,(1,1),activation='sigmoid')(lay7)
model=Model(inp,op)
model.compile(optimizer='adam',loss=dice_loss,metrics=dice_coefficient)
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')

model.fit(x_train/255,y_train/255,validation_data=(x_test/255,y_test/255),epochs=10,batch_size=8)







# inp=Input(shape=(320,320,3))
# lay1=Conv2D(32,(3,3))(inp)
# lay1=LeakyReLU(alpha=.1)(lay1)
# lay1=Conv2D(64,(3,3))(lay1)
# lay1=LeakyReLU(alpha=.1)(lay1)

# lay2=MaxPool2D(pool_size=(2,2))(lay1) # 158
# lay2=Conv2D(64,(3,3))(lay2)  # 156
# lay2=LeakyReLU(alpha=.1)(lay2)
# lay2=Conv2D(128,(3,3))(lay2) # 154
# lay2=LeakyReLU(alpha=.1)(lay2)

# lay3=MaxPool2D(pool_size=(2,2))(lay2) # 158
# lay3=Conv2D(256,(3,3))(lay3)  # 156
# lay3=LeakyReLU(alpha=.1)(lay3)
# lay3=Conv2D(512,(4,4))(lay3) # 154
# lay3=LeakyReLU(alpha=.1)(lay3)

# lay4=MaxPool2D(pool_size=(2,2))(lay3) # 158
# lay4=Conv2D(512,(3,3))(lay4)  # 156
# lay4=LeakyReLU(alpha=.1)(lay4)
# lay4=Conv2D(512,(3,3))(lay4) # 154
# lay4=LeakyReLU(alpha=.1)(lay4)

# #########################

# lay5=Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(lay4)
# crp=int(get_indices(lay3,lay5))
# lay3crp=Cropping2D((crp,crp))(lay3)
# lay5=Concatenate()([lay5,lay3crp])
# lay5=Conv2D(256,(3,3))(lay5)
# lay5=LeakyReLU(alpha=.1)(lay5)

# lay6=Conv2DTranspose(128,(3,3),strides=(2,2),padding='same')(lay5)
# crp=int(get_indices(lay2,lay6))
# lay2crp=Cropping2D((crp,crp))(lay2)
# lay6=Concatenate()([lay6,lay2crp])
# lay6=Conv2D(128,(3,3))(lay6)
# lay6=LeakyReLU(alpha=.1)(lay6)

# lay7=Conv2DTranspose(32,(3,3),strides=(2,2),padding='same')(lay6)
# crp=int(get_indices(lay1,lay7))
# lay1crp=Cropping2D((crp,crp))(lay1)
# lay7=Concatenate()([lay7,lay1crp])
# lay7=Conv2D(32,(3,3))(lay7)
# lay7=LeakyReLU(alpha=.1)(lay7)
# lay7=Conv2D(1,(1,1),activation='sigmoid')(lay7)







test=np.array(x[1])
test=np.reshape(test,(1,320,320,3))
p=model.predict(test)
cv2.imshow("predicted img",p[0,:,:,0])
cv2.imshow("true image",y[1])
cv2.imshow("actual  image",x[1])
if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()



