import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter data for the year 1933
year_1933 = df[df['year'] == '1933']

# Extract disease case values
diseases = year_1933.columns[1:]  # Exclude 'year'
cases = year_1933[diseases].values[0]  # First (and only) row of data

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(cases, labels=diseases, autopct='%1.1f%%', startangle=90)
plt.title('Proportions of Cases for Infectious Diseases in 1933')
plt.show()

# Final Answer is not a value but a visualization; however, per instructions, we print the required format
Final Answer: typhus, typhoid fever, relapsing fever, smallpox, malaria