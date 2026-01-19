import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Francisco Bravo Medical Magnet High School
school_row = df[df['School'] == 'Francisco Bravo Medical Magnet High School']
# Select the score columns from 2007 to 2011
scores = school_row[['2007', '2008', '2009', '2010', '2011']].values[0]
# Convert to float and calculate mean, ignoring None
average_score = sum(float(x) for x in scores if x is not None) / len([x for x in scores if x is not None])
print(f"Final Answer: {average_score:.1f}")