import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Outright Titles' and 'Total Finals' to numeric
df['Outright Titles'] = pd.to_numeric(df['Outright Titles'], errors='coerce')
df['Total Finals'] = pd.to_numeric(df['Total Finals'], errors='coerce')

# Filter out rows where Total Finals is 0 or NaN to avoid division by zero
valid_rows = df.dropna(subset=['Total Finals'])
valid_rows['Title Winning Percentage'] = (valid_rows['Outright Titles'] / valid_rows['Total Finals']) * 100

# Find the school with the highest percentage
max_percentage_row = valid_rows.loc[valid_rows['Title Winning Percentage'].idxmax()]
highest_percentage_school = max_percentage_row['School']

print(f"Final Answer: {highest_percentage_school}")