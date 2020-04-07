import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
import numpy as np,cv2,imutils

dataset = pd.read_csv('/home/ashika29/Downloads/train.csv')
print(dataset.head())
x=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train=x_train.reshape(33600,28,28,1)
y_train=to_categorical(y_train)

classifier=Sequential()
classifier.add(Convolution2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(10, activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,epochs=35,batch_size=500,validation_split=0.2)
classifier.save('digit_recognizer.ash')

#reading image
img = cv2.imread(x[5])
#resizing image
img = imutils.resize(img,width=300)
#showing original image
cv2.imshow("Original",img)
#converting image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#showing grayscale image
cv2.imshow("Gray Image",gray)

#creating a kernel
kernel = np.ones((40,40),np.uint8)

#applying blackhat thresholding
blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)


#applying OTSU's thresholding
ret,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#performing erosion and dilation
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

#finding countours in image
ret,cnts= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#loading our ANN model
classifier = load_model('digit_recognizer.ash')
for c in cnts:
    try:
        #creating a mask
        mask = np.zeros(gray.shape,dtype="uint8")
        
    
        (x,y,w,h) = cv2.boundingRect(c)
        
        hull = cv2.convexHull(c)
        cv2.drawContours(mask,[hull],-1,255,-1)    
        mask = cv2.bitwise_and(thresh,thresh,mask=mask)

        
        #Getting Region of interest
        roi = mask[y-7:y+h+7,x-7:x+w+7]       
        roi = cv2.resize(roi,(28,28))
        roi = np.array(roi)
        #reshaping roi to feed image to our model
        roi = roi.reshape(1,784)

        #predicting
        prediction = model.predict(roi)
        predict=prediction.argmax()
    
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(img,str(int(predict)),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),1)
        
    except Exception as e:
        print(e)
        
img = imutils.resize(img,width=500)

#showing the output
cv2.imshow('Detection',img)
