import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Male column contains percentage
male_percentage_rows = df[df['Male'].str.contains('%', na=False)]
# Convert percentage strings to float
male_percentage_rows['Male'] = male_percentage_rows['Male'].str.replace('%', '').astype(float)
# Find the row with the highest male percentage
max_male_category = male_percentage_rows.loc[male_percentage_rows['Male'].idxmax()]
print(f"Final Answer: {max_male_category['Particulars']}")