import pandas as pd

df = pd.read_csv('table.csv')
# Clean the '% of national vote' column by removing '%' and converting to float
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)
# Calculate standard deviation
std_dev = df['% of national vote'].std()
print(f"Final Answer: {std_dev:.2f}")