import pandas as pd

df = pd.read_csv('table.csv')
# Count how many conferences have a non '-' entry in the '1996' column
count_1996 = df[df['1996'] != '-'].shape[0]
print(f"Final Answer: {count_1996}")