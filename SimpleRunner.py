import os
import numpy as np
import sys
import cv2
from MyCNN import Net
from PIL import Image
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn import (
    Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax,
    BatchNorm2d, Dropout
)
from torch.optim import Adam
from MyCNN import Net
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import csv
from sklearn import preprocessing
import glob
import matplotlib.pyplot as plt


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load('best-model_MyCNN-parameters.pt'))
    model.eval()
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w, c = frame.shape
    analysisframe = ''
    letter_rgb_l = []
    letter_gray_l = []
    pixels_l = []
    le = preprocessing.LabelEncoder()
    letters=['Gamma', 'Beta', 'Eta', 'Phi', 'Theta', 'Xi', 'Zeta']
    le.fit(letters)

    while True:
        _, frame = cap.read()

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            analysisframe = frame
            showframe = analysisframe
            cv2.imshow("Frame", showframe)
            framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
            analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
            analysisframe = cv2.resize(analysisframe,(28,28))
            
            analysisframe = torch.from_numpy(analysisframe.reshape(1, 1, 28, 28)).to(device)
            analysisframe = analysisframe.float()
            device = next(model.parameters()).device
            with torch.no_grad():
                output = model(analysisframe.to(device))
            softmax = torch.exp(output)
            #prob = list(softmax.numpy())
            predictions = torch.argmax(softmax, -1)
            letter = le.inverse_transform([predictions.item()])
            print(letter)

            

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
