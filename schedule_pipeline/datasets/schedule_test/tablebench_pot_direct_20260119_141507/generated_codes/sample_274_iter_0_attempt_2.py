import pandas as pd
df = pd.read_csv('table.csv')
df['cost'] = df['cost'].str.extract(r'(\d+)').astype(float)
columns_to_check = ['2000 kwh / kwp y', '1800 kwh / kwp y', '1600 kwh / kwp y', '1400 kwh / kwp y', '1200 kwh / kwp y', '1000 kwh / kwp y', '800 kwh / kwp y']