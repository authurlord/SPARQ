import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where location contains 'austria' (case-insensitive)
austria_mountains = df[df['location'].str.contains('austria', case=False, na=False)]
count_austria = len(austria_mountains)
print(f"Final Answer: {count_austria}")