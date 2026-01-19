import pandas as pd

df = pd.read_csv('table.csv')
# Sort by year (January) to process in chronological order
df_sorted = df.sort_values(by='year (january)')
# Find the first year where 'urban , %' > 50
first_surpass_50 = df_sorted[df_sorted['urban , %'] > 50.0]['year (january)'].iloc[0]
print(f"Final Answer: {first_surpass_50}")