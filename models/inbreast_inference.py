import os

import cv2
import numpy as np
import pandas as pd
import torch
from pydicom import dcmread

dicoms_path = r'C:\Users\Korisnik\Documents\GitHub\mammography\data\INBREAST\AllDICOMs'
results_path = r'C:\Users\Korisnik\Documents\GitHub\L-CAM\datalist\inbreast\all.csv'
inbreast_csv_path = r"C:\Users\Korisnik\Documents\GitHub\mammography\data\INBREAST\INbreast.csv"

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

df = pd.read_csv(inbreast_csv_path, sep=';')

for entity in os.scandir(dicoms_path):
    mask = df['File Name'].astype(str).str.startswith(entity.name.split('_')[0])
    index = df.index[mask]
    df.loc[index, 'full_name'] = entity.name
    image = dcmread(entity.path).pixel_array
    image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255.0
    image = image.astype(np.uint8)
    image = preprocess(image)
    prediction = model.forward(image)[0][0].item()
    df.loc[index, 'prediction'] = prediction
    print(entity.name)
    print(prediction)

df.to_csv(results_path)

# BLIND DB
# df = pd.read_excel(r"C:\Users\Korisnik\Desktop\blind_db_results.xlsx")

# for index, row in df.iterrows():
#     path = os.path.join(r"C:\Users\Korisnik\Desktop\Imagebox", row['patient_id'], str(row['image_id'])+'.dcm')
#     image = dcmread(path).pixel_array
#     image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255.0
#     image = image.astype(np.uint8)
#     image = preprocess(image)
#     prediction = model.forward(image)
#     print(row['Patient'], row['Image'], row['Probability Neg.'])
#     print(prediction)
