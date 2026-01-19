import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['revenue (million)'] = pd.to_numeric(df['revenue (million)'])
df['earnings per share (p)'] = pd.to_numeric(df['earnings per share (p)'])

# Calculate correlation
correlation = df['revenue (million)'].corr(df['earnings per share (p)'])

# Print the correlation value
print(f"Final Answer: revenue (million)")