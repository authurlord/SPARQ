import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for year 1948
row_1948 = df[df['Year'] == '1948'].iloc[0]

# Extract population values (remove parentheses and convert to int)
def extract_population(value):
    if pd.isna(value) or value is None:
        return 0
    # Remove the percentage part in parentheses and convert to int
    try:
        # Find the number before the parenthesis
        num_str = value.split(' (')[0]
        return int(num_str.replace(',', ''))
    except:
        return 0

# Apply extraction to each group
serbs = extract_population(row_1948['Serbs'])
hungarians = extract_population(row_1948['Hungarians'])
germans = extract_population(row_1948['Germans'])
romanians = extract_population(row_1948['Romanians'])
slovaks = extract_population(row_1948['Slovaks'])

# Create labels and values for pie chart
labels = ['Serbs', 'Hungarians', 'Germans', 'Romanians', 'Slovaks']
values = [serbs, hungarians, germans, romanians, slovaks]

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Racial Composition of the Region in 1948')
plt.show()

# Final Answer: The pie chart is generated and displayed.
Final Answer: pie_chart_1948