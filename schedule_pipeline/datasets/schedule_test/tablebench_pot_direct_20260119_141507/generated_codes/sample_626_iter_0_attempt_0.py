import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Year built is 1909 and count them
total_locomotives_1909 = df[df['Year built'] == 1909].shape[0]
print(f"Final Answer: {total_locomotives_1909}")