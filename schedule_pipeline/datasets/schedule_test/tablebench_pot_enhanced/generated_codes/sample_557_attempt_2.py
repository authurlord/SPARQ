import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Conservative councillors' to integer
df['Conservative councillors'] = pd.to_numeric(df['Conservative councillors'])
# Calculate the difference between consecutive years
df['diff'] = df['Conservative councillors'].diff()
# Find the row with the maximum increase
max_increase_row = df.loc[df['diff'].idxmax()]
# Extract the year
year_of_max_increase = max_increase_row['Year']
print(f"Final Answer: {year_of_max_increase}")