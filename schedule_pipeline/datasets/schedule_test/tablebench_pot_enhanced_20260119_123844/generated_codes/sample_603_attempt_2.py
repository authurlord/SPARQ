import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Francisco Bravo Medical Magnet High School
school_row = df[df['School'] == 'Francisco Bravo Medical Magnet High School']
# Select the score columns from 2007 to 2011
scores = school_row[['2007', '2008', '2009', '2010', '2011']].squeeze()
# Convert to numeric and calculate mean
average_score = scores.astype(float).mean()
print(f"Final Answer: {average_score:.1f}")