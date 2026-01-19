import pandas as pd

df = pd.read_csv('table.csv')

# Extract the top 3 countries by Primary schools (excluding rows with '-' and 'Total Schools Globally', 'Countries & Territories')
top_3_primary = df[df['Country'].isin(['United States', 'Canada', 'Australia'])]['Primary'].astype(int).sum()

# Total Career-related schools from the 'Total Schools Globally' row
total_career_related = int(df[df['Country'] == 'Total Schools Globally']['Career-related'].values[0])

# Calculate the difference
difference = top_3_primary - total_career_related

print(f"Final Answer: {difference}")