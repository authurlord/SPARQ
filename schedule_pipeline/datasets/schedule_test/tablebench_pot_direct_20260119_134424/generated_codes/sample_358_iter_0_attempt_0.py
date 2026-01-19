import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'indians admitted' to integer for comparison
df['indians admitted'] = pd.to_numeric(df['indians admitted'])
# Count years with more than 25000 Indians admitted
count_years = (df['indians admitted'] > 25000).sum()
print(f"Final Answer: {count_years}")