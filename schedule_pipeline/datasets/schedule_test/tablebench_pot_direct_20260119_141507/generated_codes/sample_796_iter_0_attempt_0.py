import pandas as pd

df = pd.read_csv('table.csv')

# Filter top 5 ranked countries (rank 1 to 5)
top_5 = df[df['rank'].astype(float) <= 5].copy()

# Extract 2010 and 2011 values for the top 5 countries
values_2010 = top_5['2010'].astype(int)
values_2011 = top_5['2011'].astype(int)

# Calculate average annual growth rate (AGR) for each country
agr_2010 = ((values_2011 - values_2010) / values_2010) * 100
mean_agr_2011 = agr_2010.mean()
mean_agr_2010 = (values_2011 - values_2010) / values_2010 * 100  # same as above; just recompute for clarity
mean_agr_2010 = mean_agr_2010.mean()

# Percentage difference between mean AGR of 2011 and 2010
percentage_diff = ((mean_agr_2011 - mean_agr_2010) / mean_agr_2010) * 100

print(f"Final Answer: {percentage_diff:.2f}")