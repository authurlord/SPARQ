import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for year 1948
year_1948 = df[df['Year'] == '1948']

# Extract the ethnic groups and convert percentage strings to actual counts
total_population = 601626
ethnic_groups = ['Serbs', 'Hungarians', 'Germans', 'Romanians', 'Slovaks']
values = []

for group in ethnic_groups:
    value_str = year_1948[group].str.extract(r'(\d+\.?\d*)%')[0].fillna(0).astype(float)
    if value_str.empty:
        value_str = 0
    count = (value_str / 100) * total_population
    values.append(count)

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=ethnic_groups, autopct='%1.1f%%', startangle=90)
plt.title('Racial Composition of the Region in 1948')
plt.show()

# Final Answer is not a number or entity, but the chart is generated. Since the question asks to draw, we don't return a name.
# However, per instruction, we must follow the format. Since no specific answer name is requested, we just output the required format.
Final Answer: pie_chart_1948