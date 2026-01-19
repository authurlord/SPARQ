import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data for the year 1933
data_1933 = df[df['year'] == '1933']
# Extract disease case counts
diseases = ['typhus', 'typhoid fever', 'relapsing fever', 'smallpox', 'malaria']
cases = [int(data_1933[disease].values[0]) for disease in diseases]
# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(cases, labels=diseases, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Infectious Disease Cases in 1933')
plt.show()