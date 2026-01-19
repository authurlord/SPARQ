import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Taking Office' and 'Leaving' to integers, handling 'Current' as NaN
df['Taking Office'] = pd.to_numeric(df['Taking Office'], errors='coerce')
df['Leaving'] = pd.to_numeric(df['Leaving'], errors='coerce')

# Filter out rows where 'Leaving' is 'Current' (i.e., not a number)
df = df.dropna(subset=['Leaving'])

# Calculate years served for each mayor
df['Years Served'] = df['Leaving'] - df['Taking Office']

# Sum total years served
total_years = df['Years Served'].sum()
print(f"Final Answer: {int(total_years)}")