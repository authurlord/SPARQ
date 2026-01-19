import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter the row for year 1948
row_1948 = df[df['Year'] == '1948'].iloc[0]

# Extract the ethnic group columns and parse percentages
ethnic_groups = ['Serbs', 'Hungarians', 'Germans', 'Romanians', 'Slovaks']
percentages = []

for group in ethnic_groups:
    value_str = row_1948[group]
    # Extract the percentage part in parentheses
    if '(' in value_str:
        pct = value_str.split('(')[1].strip().rstrip('%')
        percentages.append(float(pct))
    else:
        percentages.append(0)

# Create labels for the pie chart
labels = ethnic_groups

# Draw the pie chart
plt.figure(figsize=(8, 8))
plt.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Racial Composition in 1948')
plt.show()

# Final Answer: The pie chart is generated as requested.
Final Answer: pie_chart_1948