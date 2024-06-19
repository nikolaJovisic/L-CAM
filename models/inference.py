import os

import cv2
import numpy as np
import pandas as pd
from pydicom import dcmread
import torch

from conv_model import CustomCNN, Model1

def read_img(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return preprocess(image)

def preprocess(image):
    image = cv2.resize(image, (896, 1152), interpolation=cv2.INTER_CUBIC)
    image = image.astype('float32')
    image = np.stack((image,) * 3, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.Tensor(image)
    return image

model = torch.load('inbreast_vgg16_512x1.pth')
model.eval()

df = pd.read_excel(r"C:\Users\Korisnik\Downloads\Dojka 2020-POSLATO-SVE-30.05.2024.xls")

print()



# for entity in os.scandir(r'C:\Users\Korisnik\Documents\GitHub\L-CAM\images'):
#     prediction = model.forward(image)
#     print(prediction)

# BLIND DB
# df = pd.read_excel(r"C:\Users\Korisnik\Desktop\blind_db_results.xlsx")
#
# for index, row in df.iterrows():
#     path = os.path.join(r"C:\Users\Korisnik\Desktop\Imagebox", row['patient_id'], str(row['image_id'])+'.dcm')
#     image = dcmread(path).pixel_array
#     image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255.0
#     image = image.astype(np.uint8)
#     image = preprocess(image)
#     prediction = model.forward(image)
#     print(row['Patient'], row['Image'], row['Probability Neg.'])
#     print(prediction)
