import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Outright Titles' and 'Total Finals' to numeric, handling NaN values
df['Outright Titles'] = pd.to_numeric(df['Outright Titles'], errors='coerce')
df['Total Finals'] = pd.to_numeric(df['Total Finals'], errors='coerce')

# Calculate Title Winning Percentage (only where Total Finals is not NaN)
df['Title Winning Percentage'] = (df['Outright Titles'] / df['Total Finals']) * 100

# Find the school with the highest percentage, ignoring rows where Total Finals is NaN
max_percentage_row = df.dropna(subset=['Total Finals']).loc[df['Title Winning Percentage'].idxmax()]

print(f"Final Answer: {max_percentage_row['School']}")