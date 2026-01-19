import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows with male data and extract male percentage or count
male_percentage = df[df['Male'] != '-']['Male']
# Convert to numeric if possible, otherwise treat as string
male_values = []
for val in male_percentage:
    if isinstance(val, str) and '%' in val:
        male_values.append(float(val.replace('%', '')))
    else:
        male_values.append(float(val))

# Find the maximum male percentage
max_male_value = max(male_values)
# Get the corresponding category
category = df[df['Male'].astype(str).str.replace('%', '').str.strip() == str(max_male_value)][
    'Particulars'].values[0]

print(f"Final Answer: {category}")