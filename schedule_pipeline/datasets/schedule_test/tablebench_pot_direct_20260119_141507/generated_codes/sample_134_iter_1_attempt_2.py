import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter data for year 1933
year_1933 = df[df['year'] == '1933']

# Extract disease case counts
diseases = ['typhus', 'typhoid fever', 'relapsing fever', 'smallpox', 'malaria']
cases = year_1933[diseases].values[0]  # Get the first (and only) row's values

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(cases, labels=diseases, autopct='%1.1f%%', startangle=90)
plt.title('Proportions of Cases for Infectious Diseases in 1933')
plt.show()

# Final Answer: The pie chart is displayed above.
Final Answer: pie_chart