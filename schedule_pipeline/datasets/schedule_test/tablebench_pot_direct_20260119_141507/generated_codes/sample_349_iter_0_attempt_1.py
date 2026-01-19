import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of conferences with occurrences in 1996 (non '-' values in the '1996' column)
count_1996 = df[df['1996'] != '-'].shape[0]
print(f"Final Answer: {count_1996}")