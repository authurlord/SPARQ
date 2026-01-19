import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter data for the year 1948
row_1948 = df[df['Year'] == '1948']

# Extract the ethnic groups and their percentages
groups = ['Serbs', 'Hungarians', 'Germans', 'Romanians', 'Slovaks']
percentages = []

for group in groups:
    value_str = row_1948[group].values[0]
    if pd.isna(value_str):
        percentages.append(0)
    else:
        # Extract the numeric part from the string like "59.6%"
        percentage = float(value_str.split('(')[1].strip('%)'))
        percentages.append(percentage)

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(percentages, labels=groups, autopct='%1.1f%%', startangle=90)
plt.title('Racial Composition in 1948')
plt.show()

print("Final Answer: Serbs, Hungarians, Germans, Romanians, Slovaks")