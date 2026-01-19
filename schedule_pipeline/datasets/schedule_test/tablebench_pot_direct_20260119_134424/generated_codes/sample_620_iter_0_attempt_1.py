import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2002 to 2005
filtered_df = df[df['year'].isin(['2002', '2004', '2005', '2006'])]
# Convert mintage to integer and sum
total_mintage = filtered_df['mintage'].astype(int).sum()
# Divide by 5 people
coins_per_person = total_mintage // 5
print(f"Final Answer: {coins_per_person}")