import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Francisco Bravo Medical Magnet High School
school_data = df[df['School'] == 'Francisco Bravo Medical Magnet High School']
# Extract the scores for years 2007 to 2011 and convert to float
scores = [float(x) if x else None for x in school_data[['2007', '2008', '2009', '2010', '2011']].values[0]]
# Calculate the mean, ignoring NaN
mean_score = sum(score for score in scores if score) / sum(1 for score in scores if score)
print(f"Final Answer: {mean_score:.1f}")