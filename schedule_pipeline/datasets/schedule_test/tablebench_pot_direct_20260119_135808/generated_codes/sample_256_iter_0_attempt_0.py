import pandas as pd

df = pd.read_csv('table.csv')

# Extract the top 3 countries by Primary schools
top_3_primary = df.iloc[0:3]['Primary'].astype(int).sum()

# Extract total Career-related schools from the "Total Schools Globally" row
total_career_related = df.iloc[-3]['Career-related']

# Calculate the difference
difference = top_3_primary - int(total_career_related)

print(f"Final Answer: {difference}")