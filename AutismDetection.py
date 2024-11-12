from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from keras.applications import ResNet50
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
import cv2
import os
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.applications import xception
import webbrowser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd

main = tkinter.Tk()
main.title("Autism Detection using Resnet50 & Xception Transfer Learning") 
main.geometry("1300x1200")

global filename, resnet_model, X, Y
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore 
labels = ['Autistic', 'Non_Autistic']

#function to return integer label for given image
def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name: #return integer ID as label for given plant disease name
            index = i
            break
    return index


def uploadDataset(): 
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");


def preprocess():
    global filename, cnn, X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy') #load X and Y data
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j]) #read image
                    img = cv2.resize(img, (64,64)) #resize image
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(64,64,3) #resize as colur image
                    label = getID(name) #get id or label 
                    X.append(im2arr) #add all image pixel to X array
                    Y.append(label) #add label to Y array
                    print(name+" "+root+"/"+directory[j]+" "+str(label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X) #save X and Y data for future user
        np.save('model/Y.txt',Y)
    X = X.astype('float32') #normalize image pixel with float values
    X = X/255
    indices = np.arange(X.shape[0]) #shuffling the images
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset Preprocessing & Image Normalization Process Completed\n\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% images used to train Resnet50 & Xception algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% images used to test Resnet50 & Xception algorithms : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    test = X[3]
    cv2.imshow("Processed Image",cv2.resize(test,(200,200)))
    cv2.waitKey(0)
       
def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runResnet():
    global X_train, X_test, y_train, y_test, resnet_model
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    #creating resnet object
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]))
    resnet_model = Sequential()
    #transfering resnet50 object to custom cnn model
    resnet_model.add(resnet)
    resnet_model.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (1, 1)))
    resnet_model.add(Convolution2D(32, 1, 1, activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (1, 1)))
    resnet_model.add(Flatten())
    resnet_model.add(Dense(output_dim = 256, activation = 'relu'))
    resnet_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #compiling custome model
    if os.path.exists('model/resnet_model_weights.hdf5') == False:
        model_check_point = ModelCheckpoint(filepath='model/resnet_model_weights.hdf5', verbose = 1, save_best_only = True)
        #training the model
        hist = resnet_model.fit(X_train, y_train, batch_size = 16, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/resnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        resnet_model = load_model('model/resnet_model_weights.hdf5')
    predict = resnet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    target = np.argmax(y_test, axis=1)
    calculateMetrics("Resnet50", predict, target)

def runXception():
    #creating xception object
    xcept = xception.Xception(weights='imagenet', include_top=False, input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]))
    xception_model = Sequential()
    #transfering xception object to custom cnn model
    xception_model.add(xcept)
    xception_model.add(Convolution2D(32, kernel_size=(3, 3),strides=(3, 3), input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]), activation = 'relu'))
    xception_model.add(MaxPooling2D(pool_size = (2, 2), padding="same"))
    xception_model.add(Convolution2D(32, kernel_size=(3, 3),strides=(3, 3), activation = 'relu', padding="same"))
    xception_model.add(MaxPooling2D(pool_size = (2, 2), padding="same"))
    xception_model.add(Flatten())
    xception_model.add(Dense(output_dim = 256, activation = 'relu'))
    xception_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    xception_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists('model/xception_model_weights.hdf5') == False:
        model_check_point = ModelCheckpoint(filepath='model/xception_model_weights.hdf5', verbose = 1, save_best_only = True)
        hist = xception_model.fit(X_train, y_train, batch_size = 16, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/resnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        xception_model = load_model('model/xception_model_weights.hdf5')
    predict = xception_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    target = np.argmax(y_test, axis=1)
    calculateMetrics("Xception", predict, target)    

def graph():
    output = "<html><body><table align=center border=1><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th>"
    output+="<th>FSCORE</th></tr>"
    output+="<tr><td>Resnet 50</td><td>"+str(accuracy[0])+"</td><td>"+str(precision[0])+"</td><td>"+str(recall[0])+"</td><td>"+str(fscore[0])+"</td></tr>"
    output+="<tr><td>Xception</td><td>"+str(accuracy[1])+"</td><td>"+str(precision[1])+"</td><td>"+str(recall[1])+"</td><td>"+str(fscore[1])+"</td></tr>"
    output+="</table></body></html>"
    f = open("table.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("table.html",new=2)
    
    df = pd.DataFrame([['Resnet50','Precision',precision[0]],['Resnet50','Recall',recall[0]],['Resnet50','F1 Score',fscore[0]],['Resnet50','Accuracy',accuracy[0]],
                       ['Xception','Precision',precision[1]],['Xception','Recall',recall[1]],['Xception','F1 Score',fscore[1]],['Xception','Accuracy',accuracy[1]],
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()


def predict():
    global resnet_model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (80, 80))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,80, 80,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = resnet_model.predict(img)
    predict = np.argmax(preds)
    score = np.amax(preds)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Classification Result: '+labels[predict]+" Detected", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Classification Result: '+labels[predict]+" Detected", img)
    cv2.waitKey(0)


font = ('times', 16, 'bold')
title = Label(main, text='Autism Detection using Resnet50 & Xception Transfer Learning')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Autism Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocess, bg='#ffb3fe')
processButton.place(x=340,y=550)
processButton.config(font=font1) 

resnetButton1 = Button(main, text="Run Resnet50 Algorithm", command=runResnet, bg='#ffb3fe')
resnetButton1.place(x=570,y=550)
resnetButton1.config(font=font1) 

xceptionButton = Button(main, text="Run Xception Algorithm", command=runXception, bg='#ffb3fe')
xceptionButton.place(x=50,y=600)
xceptionButton.config(font=font1) 

graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=340,y=600)
graphButton.config(font=font1) 

predictButton = Button(main, text="Predict Autism from Test Image", command=predict, bg='#ffb3fe')
predictButton.place(x=570,y=600)
predictButton.config(font=font1) 

main.config(bg='LightSalmon3')
main.mainloop()
