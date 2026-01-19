import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'screens' column to numeric
df['screens'] = pd.to_numeric(df['screens'])
# Select top 5 chains (first 5 rows)
top_5_screens = df['screens'].head(5)
# Calculate standard deviation
std_screens = top_5_screens.std()
print(f"Final Answer: {std_screens:.2f}")