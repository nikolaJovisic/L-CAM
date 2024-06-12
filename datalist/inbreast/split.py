import random

import pandas as pd

df = pd.read_csv(r"all.csv")
entries = []
for i, row in df.iterrows():
    if row['Bi-Rads'] in ['1', '2'] and float(row['prediction']) > 0.5:
        entries.append((row['full_name'], '0'))
    if row['Bi-Rads'] in ['4', '4a', '4b', '4c', '5', '6'] and float(row['prediction']) < 0.5:
        entries.append((row['full_name'], '1'))
random.seed(42)
random.shuffle(entries)

train = entries[:270]
validation = entries[270:]

with open(r"C:\Users\Korisnik\Documents\GitHub\L-CAM\datalist\inbreast\train.txt", 'a') as file:
    for filename, label in train:
        file.write(f'{filename} {label}\n')

with open(r"C:\Users\Korisnik\Documents\GitHub\L-CAM\datalist\inbreast\validation.txt", 'a') as file:
    for filename, label in validation:
        file.write(f'{filename} {label}\n')
