import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter out the summary rows (last few rows)
# The actual country data is from index 0 to 15, excluding the last two summary rows
country_data = df.iloc[0:15]  # First 15 rows (0 to 14)

# Extract relevant columns
primary_schools = country_data['Primary'].astype(int)
career_related_schools = df.loc[df['Country'] == 'Total Schools Globally', 'Career-related'].values[0]

# Find top 3 countries by Primary schools
top_3_primary = primary_schools.nlargest(3)
sum_top_3_primary = top_3_primary.sum()

# Total career-related schools globally (from the "Total Schools Globally" row)
total_career_related = int(career_related_schools)

# Calculate the difference
difference = sum_top_3_primary - total_career_related

print(f"Final Answer: {difference}")