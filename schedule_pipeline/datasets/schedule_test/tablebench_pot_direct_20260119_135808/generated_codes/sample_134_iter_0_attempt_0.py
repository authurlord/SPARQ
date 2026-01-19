import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data for the year 1933
data_1933 = df[df['year'] == '1933']
# Select disease columns
diseases = data_1933[['typhus', 'typhoid fever', 'relapsing fever', 'smallpox', 'malaria']].values.flatten()
# Disease names
labels = ['Typhus', 'Typhoid Fever', 'Relapsing Fever', 'Smallpox', 'Malaria']
# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(diseases, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Infectious Disease Cases in 1933')
plt.show()