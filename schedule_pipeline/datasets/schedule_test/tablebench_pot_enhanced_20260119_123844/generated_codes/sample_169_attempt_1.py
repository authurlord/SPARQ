import pandas as pd

df = pd.read_csv('table.csv')

# Convert numerical columns to float
numerical_columns = ['revenue (million)', 'profit / (loss) before tax (m)', 'net profit (m)', 'earnings per share (p)']
df[numerical_columns] = df[numerical_columns].astype(float)

# Compute correlation with 'earnings per share (p)'
correlations = df[numerical_columns].corr()['earnings per share (p)'].abs().sort_values(ascending=False)

# Get the most influential factor (excluding 'earnings per share (p)' itself)
most_influential = correlations.index[1]  # Second highest (first is self-correlation)

print(f"Final Answer: {most_influential}")