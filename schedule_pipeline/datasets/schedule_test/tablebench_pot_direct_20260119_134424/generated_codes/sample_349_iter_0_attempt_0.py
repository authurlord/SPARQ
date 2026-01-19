import pandas as pd

df = pd.read_csv('table.csv')
# Select only the columns related to years, starting from '1993' to '2015'
year_columns = ['1993', '1996', '1999', '-', '2007', '2011', '2015']
# Filter rows where '1996' is not '-' (i.e., conference occurred in 1996)
count_1996 = df[df['1996'] != '-'].shape[0]
print(f"Final Answer: {count_1996}")