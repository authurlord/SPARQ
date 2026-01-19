import pandas as pd

df = pd.read_csv('table.csv')

# Extract the top 3 countries by Primary schools
top_3_primary = df.iloc[0:3]['Primary'].astype(int).sum()

# Total Career-related schools globally
total_career_related = int(df.iloc[-2]['Career-related'])

# Calculate difference
difference = top_3_primary - total_career_related

print(f"Final Answer: {difference}")