import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract relevant columns
population = df['metropolitan population (2006) millions']
gdp_per_capita = df['gdp (ppp) us per capita']

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(population, gdp_per_capita, color='blue')
plt.title('GDP (PPP) per Capita vs Metropolitan Population')
plt.xlabel('Metropolitan Population (millions)')
plt.ylabel('GDP (PPP) per Capita (US$)')
plt.grid(True)
plt.show()

# Final Answer: Based on the scatter plot, there is no clear linear trend; GDP per capita varies independently of population size.
Final Answer: No clear trend