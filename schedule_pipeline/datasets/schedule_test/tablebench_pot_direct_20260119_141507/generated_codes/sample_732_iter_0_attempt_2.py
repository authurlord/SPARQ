import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Male' has a percentage value (non-empty and not '-')
male_percentage_rows = df[df['Male'] != '-']
# Convert 'Male' column to numeric for comparison
male_percentage = pd.to_numeric(male_percentage_rows['Male'], errors='coerce')
# Find the row with the highest male percentage
max_male_row = male_percentage.idxmax()
# Get the corresponding 'Particulars' value
highest_male_category = df.loc[max_male_row, 'Particulars']
print(f"Final Answer: {highest_male_category}")