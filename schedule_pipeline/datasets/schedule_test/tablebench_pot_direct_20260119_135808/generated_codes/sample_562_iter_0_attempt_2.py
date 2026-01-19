import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2002 to 2006
filtered_df = df[df['year'].astype(int).between(2002, 2006)]
# Calculate total mintage
total_mintage = filtered_df['mintage'].astype(int).sum()
print(f"Final Answer: {total_mintage}")