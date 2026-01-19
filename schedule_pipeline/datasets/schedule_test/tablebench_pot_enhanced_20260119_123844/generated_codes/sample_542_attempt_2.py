import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 2004
mintage_2004 = df[df['year'] == '2004']['mintage'].astype(int).sum()
print(f"Final Answer: {mintage_2004}")