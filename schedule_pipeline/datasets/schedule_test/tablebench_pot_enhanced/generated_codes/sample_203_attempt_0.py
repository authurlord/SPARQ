import pandas as pd

df = pd.read_csv('table.csv')
# Drop rows where rating is 'tba'
df = df[df['rating'] != 'tba']
# Convert 'viewers (m)' and 'rating' to numeric
df['viewers (m)'] = pd.to_numeric(df['viewers (m)'])
df['rating'] = pd.to_numeric(df['rating'])
# Calculate correlation
correlation = df['viewers (m)'].corr(df['rating'])
print(f"Final Answer: {correlation:.2f}")