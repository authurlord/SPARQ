import pandas as pd

df = pd.read_csv('table.csv')
# Display the column names and data types
print("Key columns:", df.columns.tolist())
print("Data types:", df.dtypes.to_dict())
print("Summary statistics:")
print(df.describe(include='all').to_string())