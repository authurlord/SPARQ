import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Male column contains percentage (i.e., literacy)
literacy_row = df[df['Male'].str.contains('%', na=False)]
# Convert Male percentage to float and find max
max_male_percentage = literacy_row['Male'].str.replace('%', '').astype(float).max()
# Get the corresponding Particulars
category = literacy_row[literacy_row['Male'].str.replace('%', '').astype(float) == max_male_percentage]['Particulars'].iloc[0]
print(f"Final Answer: {category}")