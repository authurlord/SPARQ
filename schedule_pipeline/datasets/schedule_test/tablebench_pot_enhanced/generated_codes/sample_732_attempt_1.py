import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Male column has percentage values (not '-' or numeric)
male_percentages = df[df['Male'].str.contains('%', na=False)]
# Convert percentage strings to float for comparison
male_percentages['Male'] = male_percentages['Male'].str.replace('%', '').astype(float)
# Find the row with the highest male percentage
max_male_row = male_percentages.loc[male_percentages['Male'].idxmax()]
print(f"Final Answer: {max_male_row['Particulars']}")