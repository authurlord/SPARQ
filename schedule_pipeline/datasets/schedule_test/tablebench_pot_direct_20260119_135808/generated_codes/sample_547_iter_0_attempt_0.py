import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Ghent
ghent_data = df[df['location'] == 'ghent']

# Group by year and calculate average final score
avg_scores = ghent_data.groupby('year')['score - final'].mean()

# Extract scores for 2010 and 2011
score_2010 = avg_scores['2010']
score_2011 = avg_scores['2011']

# Calculate difference
difference = score_2011 - score_2010
print(f"Final Answer: {difference:.3f}")