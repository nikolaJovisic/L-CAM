import pandas as pd

inbreast_csv_path = r"C:\Users\Korisnik\Documents\GitHub\mammography\data\INBREAST\INbreast.csv"
df = pd.read_csv(inbreast_csv_path, sep=';')
with open(r"C:\Users\Korisnik\Documents\GitHub\mammography\data\INBREAST\results.txt", 'r') as file:
    while True:
        name = file.readline()[:-1]
        if not name:
            break
        value = file.readline()[:-1]
        mask = df['File Name'].astype(str).str.startswith(name.split('_')[0])
        index = df.index[mask]
        df.loc[index, 'full_name'] = name
        df.loc[index, 'prediction'] = value


results_path = r'C:\Users\Korisnik\Documents\GitHub\L-CAM\datalist\inbreast\all.csv'
df.to_csv(results_path)