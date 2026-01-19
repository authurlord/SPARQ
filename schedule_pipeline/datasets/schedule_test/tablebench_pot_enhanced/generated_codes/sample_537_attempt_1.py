import pandas as pd

df = pd.read_csv('table.csv')
# Find the first year where urban percentage exceeds 50%
urban_surpass = df[df['urban , %'].astype(int) > 50]['year (january)'].iloc[0]
print(f"Final Answer: {urban_surpass}")