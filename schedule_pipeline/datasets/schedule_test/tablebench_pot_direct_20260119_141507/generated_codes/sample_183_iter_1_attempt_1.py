import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean the net profit and basic eps columns by removing the leading '-' and space
df['net profit / loss (sek)'] = df['net profit / loss (sek)'].str.replace('- ', '', regex=False).astype(float)
df['basic eps (sek)'] = df['basic eps (sek)'].str.replace('- ', '', regex=False).astype(float)

# Select relevant columns for analysis
columns_to_analyze = ['passengers flown', 'employees (average / year)', 'basic eps (sek)']
target_column = 'net profit / loss (sek)'

# Convert 'passengers flown' and 'employees' to numeric (they are already strings with numbers)
df['passengers flown'] = pd.to_numeric(df['passengers flown'], errors='coerce')
df['employees (average / year)'] = pd.to_numeric(df['employees (average / year)'], errors='coerce')

# Drop rows with NaN after conversion
df = df.dropna()

# Compute correlation with net profit / loss (sek)
correlations = df[columns_to_analyze].corrwith(df[target_column])

# Identify which factors have a significant correlation (absolute value > 0.3)
significant_factors = []
for col in columns_to_analyze:
    if abs(correlations[col]) > 0.3:
        significant_factors.append(col)

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")