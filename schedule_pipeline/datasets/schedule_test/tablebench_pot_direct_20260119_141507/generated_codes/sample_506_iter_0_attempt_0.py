import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Outright Titles' and 'Total Finals' to numeric, handling any potential string issues
df['Outright Titles'] = pd.to_numeric(df['Outright Titles'], errors='coerce')
df['Total Finals'] = pd.to_numeric(df['Total Finals'], errors='coerce')

# Calculate Title Winning Percentage, skipping rows where Total Finals is 0 or NaN
df['Title Winning Percentage'] = (df['Outright Titles'] / df['Total Finals']) * 100

# Find the school with the highest percentage
max_percentage_row = df.loc[df['Title Winning Percentage'].idxmax()]
highest_percentage_school = max_percentage_row['School']

print(f"Final Answer: {highest_percentage_school}")