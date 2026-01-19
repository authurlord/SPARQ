import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2011 (imf)' column to integer and calculate the sum
total_gdp_2011 = df['2011 (imf)'].astype(int).sum()
print(f"Final Answer: {total_gdp_2011}")