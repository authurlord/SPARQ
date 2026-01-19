import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['pop (2010)'] = pd.to_numeric(df['pop (2010)'])
df['land ( sqmi)'] = pd.to_numeric(df['land ( sqmi)'])

# Calculate correlation coefficient
correlation = df['pop (2010)'].corr(df['land ( sqmi)'])
print(f"Final Answer: {correlation:.4f}")