import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter the row for year 1948
year_1948 = df[df['Year'] == '1948'].iloc[0]

# Extract the ethnic groups and clean the values
ethnic_groups = ['Serbs', 'Hungarians', 'Germans', 'Romanians', 'Slovaks']
values = []

for group in ethnic_groups:
    val_str = year_1948[group]
    if pd.isna(val_str):
        values.append(0)
        continue
    # Remove the percentage part and keep only the number
    # Split by '(' and take the first part
    num_part = val_str.split('(')[0].strip()
    # Replace comma with empty string and convert to int
    clean_val = num_part.replace(',', '')
    values.append(int(clean_val))

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=ethnic_groups, autopct='%1.1f%%', startangle=90)
plt.title('Racial Composition of the Region in 1948')
plt.show()

# Final Answer: The pie chart has been generated showing the racial composition.
Final Answer: pie_chart