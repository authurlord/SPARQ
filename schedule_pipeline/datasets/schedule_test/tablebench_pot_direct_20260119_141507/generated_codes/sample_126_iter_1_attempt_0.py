import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter for year 1948
year_1948 = df[df['Year'] == '1948']

# Extract the ethnic group values by removing the text after the parenthesis
groups = ['Serbs', 'Hungarians', 'Germans', 'Romanians', 'Slovaks']
values = {}

for group in groups:
    value_str = year_1948[group].str.extract(r'(\d+,\d+)').fillna(0)
    # Convert to integer, replacing commas with dots and then convert
    value_clean = value_str.apply(lambda x: int(x[0].replace(',', '')) if pd.notna(x[0]) else 0)
    values[group] = value_clean.iloc[0]

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(values.values(), labels=values.keys(), autopct='%1.1f%%', startangle=90)
plt.title('Racial Composition of the Region in 1948')
plt.show()

# Final Answer: The pie chart is generated showing the racial composition
Final Answer: pie_chart