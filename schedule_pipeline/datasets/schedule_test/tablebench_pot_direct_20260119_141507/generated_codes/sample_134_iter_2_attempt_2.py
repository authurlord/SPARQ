import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter the data for the year 1933
year_1933 = df[df['year'] == '1933']

# Extract the disease case counts
diseases = ['typhus', 'typhoid fever', 'relapsing fever', 'smallpox', 'malaria']
cases = year_1933[diseases].values[0]  # Get the first (and only) row's values

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(cases, labels=diseases, autopct='%1.1f%%', startangle=90)
plt.title('Proportions of Cases for Infectious Diseases in 1933')
plt.show()

# Final Answer: The pie chart is displayed, so no numerical answer is needed.
Final Answer: pie_chart_1933