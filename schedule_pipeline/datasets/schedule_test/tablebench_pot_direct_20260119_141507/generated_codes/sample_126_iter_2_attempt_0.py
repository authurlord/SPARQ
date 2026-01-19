import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter data for year 1948
year_1948 = df[df['Year'] == '1948']

# Extract the ethnic group percentages (remove the % sign and convert to float)
groups = ['Serbs', 'Hungarians', 'Germans', 'Romanians', 'Slovaks']
percentages = []

for group in groups:
    value_str = year_1948[group].str.strip().str.extract(r'\((\d+(\.\d+)?)%\)$')[0].fillna(0)
    percentages.append(float(value_str))

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(percentages, labels=groups, autopct='%1.1f%%', startangle=90)
plt.title('Racial Composition in 1948')
plt.show()