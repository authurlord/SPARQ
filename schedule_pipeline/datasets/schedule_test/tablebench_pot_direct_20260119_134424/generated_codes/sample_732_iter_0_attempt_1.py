import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Male' contains a percentage (i.e., literacy)
male_percentage_rows = df[df['Male'].str.contains('%', na=False)]
# Find the row with the highest male percentage
max_male_percentage_row = male_percentage_rows.loc[male_percentage_rows['Male'].str.replace('%', '').astype(float).idxmax()]
print(f"Final Answer: {max_male_percentage_row['Particulars']}")