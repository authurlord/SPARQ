import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for year 1948
row_1948 = df[df['Year'] == '1948'].iloc[0]

# Extract the percentage values (remove the text part and convert to float)
columns = ['Serbs', 'Hungarians', 'Germans', 'Romanians', 'Slovaks']
percentages = {}

for col in columns:
    value_str = row_1948[col]
    # Extract the percentage part (e.g., '59.6%')
    if '(' in value_str:
        percent_part = value_str.split('(')[1].strip().rstrip('%')
        percentages[col] = float(percent_part)

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(percentages.values(), labels=percentages.keys(), autopct='%1.1f%%', startangle=90)
plt.title('Racial Composition in 1948')
plt.show()