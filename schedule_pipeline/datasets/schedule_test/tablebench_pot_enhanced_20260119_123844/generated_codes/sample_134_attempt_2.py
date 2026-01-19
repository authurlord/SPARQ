import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data for the year 1933
data_1933 = df[df['year'] == '1933']
# Extract disease names and case counts
diseases = data_1933.columns[1:]  # Exclude 'year'
cases = data_1933.iloc[0, 1:].astype(int)
# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(cases, labels=diseases, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Infectious Disease Cases in 1933')
plt.show()