import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, handling missing values
df['Outright Titles'] = pd.to_numeric(df['Outright Titles'], errors='coerce')
df['Total Finals'] = pd.to_numeric(df['Total Finals'], errors='coerce')

# Calculate Title Winning Percentage
df['Title Winning Percentage'] = (df['Outright Titles'] / df['Total Finals']) * 100

# Find the school with the highest Title Winning Percentage
max_percentage_school = df.loc[df['Title Winning Percentage'].idxmax(), 'School']

print(f"Final Answer: {max_percentage_school}")