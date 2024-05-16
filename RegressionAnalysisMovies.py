
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
# Load the dataset
df = pd.read_csv("imdb_movies.csv")

# Convert 'date_x' to datetime to extract the year, handling parsing errors
df['date_x'] = pd.to_datetime(df['date_x'], errors='coerce')
df['year'] = df['date_x'].dt.year

# Filter for movies from 2014 onwards
df_filtered = df[df['year'] >= 2014]

# Select only the relevant columns: names, year, score (rating), revenue, and budget_x (budget)
cleaned_df = df_filtered[['names', 'year', 'score', 'revenue', 'budget_x']]

# Optionally, save the cleaned dataset to a new CSV file
cleaned_df.to_csv('cleaned_movies_dataset.csv', index=False)

print(cleaned_df.head())  # Display the first few rows of the cleaned dataset
# Ensure 'score' and 'budget_x' are numerical and 'revenue' is float for the correlation
cleaned_df['score'] = pd.to_numeric(cleaned_df['score'], errors='coerce')
cleaned_df['budget_x'] = pd.to_numeric(cleaned_df['budget_x'], errors='coerce')
cleaned_df['revenue'] = cleaned_df['revenue'].astype(float)

# Drop rows with missing values
cleaned_df.dropna(subset=['score', 'revenue', 'budget_x'], inplace=True)

# Scatter plot for IMDb rating vs Revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='score', y='revenue', data=cleaned_df)
plt.title('IMDb Rating vs Revenue')
plt.xlabel('IMDb Rating')
plt.ylabel('Revenue')
plt.show()  # Will display the plot

# Scatter plot for Budget vs Revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='budget_x', y='revenue', data=cleaned_df)
plt.title('Production Budget vs Revenue')
plt.xlabel('Production Budget')
plt.ylabel('Revenue')
plt.show()  # Will display the plot
plt.savefig()
# Calculate and print Pearson's correlation coefficient and p-value for IMDb Rating and Revenue
rating_corr, rating_p_value = pearsonr(cleaned_df['score'], cleaned_df['revenue'])
print(f"Correlation coefficient (Rating vs Revenue): {rating_corr}")
print(f"P-value (Rating vs Revenue): {rating_p_value}")

# Calculate and print Pearson's correlation coefficient and p-value for Budget and Revenue
budget_corr, budget_p_value = pearsonr(cleaned_df['budget_x'], cleaned_df['revenue'])
print(f"Correlation coefficient (Budget vs Revenue): {budget_corr}")
print(f"P-value (Budget vs Revenue): {budget_p_value}")
