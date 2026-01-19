import pandas as pd

df = pd.read_csv('table.csv')
# Drop the first row which contains the component description
df_clean = df.drop(0)
# Find the index of the maximum value in 'Copper (mg)' column
max_index = df_clean['Copper (mg)'].idxmax()
# Get the staple food name from the 'STAPLE:' column
highest_copper_staple = df_clean.loc[max_index, 'STAPLE:']
print(f"Final Answer: {highest_copper_staple}")