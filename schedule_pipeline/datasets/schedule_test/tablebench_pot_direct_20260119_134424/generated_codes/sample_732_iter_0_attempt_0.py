import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows with male data and convert male percentage to float
male_percentages = {
    row['Particulars']: float(row['Male'].strip('%')) 
    for _, row in df.iterrows() 
    if row['Male'] != '-' and row['Male'] != '0' and '%' in row['Male']
}

# Find the category with the highest male percentage
max_male_category = max(male_percentages, key=male_percentages.get)
print(f"Final Answer: {max_male_category}")