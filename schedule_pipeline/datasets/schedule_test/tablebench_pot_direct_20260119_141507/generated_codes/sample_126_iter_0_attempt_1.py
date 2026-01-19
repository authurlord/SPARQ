import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter the row for year 1948
row_1948 = df[df['Year'] == '1948']

# Extract the ethnic group columns and remove the percentage text
ethnic_groups = ['Serbs', 'Hungarians', 'Germans', 'Romanians', 'Slovaks']
values = []

for group in ethnic_groups:
    value_str = row_1948[group].str.strip().str.extract(r'(\d+,\d+)').fillna(0)
    # Convert string with comma to integer
    if not value_str.empty:
        value = int(value_str[0].str.replace(',', '').astype(str))
    else:
        value = 0
    values.append(value)

# Create labels for the pie chart
labels = ethnic_groups

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Racial Composition of the Region in 1948')
plt.show()