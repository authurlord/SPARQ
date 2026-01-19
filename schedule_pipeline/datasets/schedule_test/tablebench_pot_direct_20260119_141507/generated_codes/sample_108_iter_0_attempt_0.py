import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the 'Gross' column and clean the values
df['Gross'] = df['Gross'].str.replace('$', '').str.replace(',', '').astype(float)

# Take the top 10 movies by rank
top_10 = df.head(10)

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_10['Gross'], labels=top_10['Title'], autopct='%1.1f%%', startangle=90)
plt.title('Box Office Earnings of Top 10 Ranked Movies')
plt.show()

# Final Answer is not a value but a chart, so we just print a placeholder indicating completion
Final Answer: pie_chart_completed