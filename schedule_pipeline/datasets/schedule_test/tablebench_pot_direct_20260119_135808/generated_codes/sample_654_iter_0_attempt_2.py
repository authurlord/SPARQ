import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'IP' and 'SO' to numeric
df['IP'] = pd.to_numeric(df['IP'], errors='coerce')
df['SO'] = pd.to_numeric(df['SO'], errors='coerce')

# Calculate correlation coefficient between IP and SO
correlation = df['IP'].corr(df['SO'])
print(f"Final Answer: {correlation:.3f}")