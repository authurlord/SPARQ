import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract the required columns
population = df['metropolitan population (2006) millions']
gdp_per_capita = df['gdp (ppp) us per capita']

# Plot scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
plt.scatter(population, gdp_per_capita, color='blue')
plt.title('GDP (PPP) per Capita vs Metropolitan Population')
plt.xlabel('Metropolitan Population (millions)')
plt.ylabel('GDP (PPP) per Capita (US$)')
plt.grid(True)
plt.show()

# Calculate correlation coefficient to quantify the relationship
correlation = population.corr(gdp_per_capita)
print(f"Final Answer: {correlation:.2f}")