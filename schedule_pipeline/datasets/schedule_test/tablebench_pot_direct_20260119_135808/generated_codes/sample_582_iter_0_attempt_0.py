import pandas as pd

df = pd.read_csv('table.csv')
# Convert percentage column to float
df['Percentage (based on registration by the church itself)'] = df['Percentage (based on registration by the church itself)'].astype(float)
# Calculate the difference between consecutive years
df['change'] = df['Percentage (based on registration by the church itself)'].diff()
# Find the year with the maximum decrease (most negative change)
max_decrease_year = df.loc[df['change'].idxmin(), 'year']
print(f"Final Answer: {max_decrease_year}")