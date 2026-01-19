import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 chains
top_5 = df[df['rank'].astype(int) <= 5]
# Convert 'screens' to numeric
top_5['screens'] = pd.to_numeric(top_5['screens'])
# Calculate standard deviation
std_screens = top_5['screens'].std()
print(f"Final Answer: {std_screens:.2f}")