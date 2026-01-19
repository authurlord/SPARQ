import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for year 1948
row_1948 = df[df['Year'] == '1948']

# Extract the relevant columns (population values)
serbs = float(row_1948['Serbs'].str.extract(r'(\d+,\d+)').values[0].replace(',', ''))
hungarians = float(row_1948['Hungarians'].str.extract(r'(\d+,\d+)').values[0].replace(',', ''))
germans = float(row_1948['Germans'].str.extract(r'(\d+,\d+)').values[0].replace(',', ''))
romanians = row_1948['Romanians'].values[0]  # This is None, so skip
slovaks = float(row_1948['Slovaks'].str.extract(r'(\d+,\d+)').values[0].replace(',', ''))

# Since Romanians is None, we exclude it
data = [serbs, hungarians, germans, slovaks]
labels = ['Serbs', 'Hungarians', 'Germans', 'Slovaks']

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Racial Composition of the Region in 1948')
plt.show()

# Final Answer: The pie chart has been generated showing the racial composition.
Final Answer: pie_chart_1948