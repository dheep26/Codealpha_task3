# Titanic Data Visualization - Task 3 (CodeAlpha Internship)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the visual theme
sns.set(style="darkgrid")

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Print basic info
print("Basic Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# -----------------------------------------
# Plot 1: Count of Survivors
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=df, palette='Set2')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Passenger Count')
plt.savefig('assets/survival_count.png')
plt.show()

# -----------------------------------------
# Plot 2: Survival by Sex
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', hue='survived', data=df, palette='Set1')
plt.title('Survival by Gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.savefig('assets/survival_by_sex.png')
plt.show()

# -----------------------------------------
# Plot 3: Age Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['age'].dropna(), kde=True, bins=30, color='teal')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('assets/age_distribution.png')
plt.show()

# -----------------------------------------
# Plot 4: Passenger Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='pclass', data=df, palette='muted')
plt.title('Passenger Class Distribution')
plt.xlabel('Class (1 = Upper, 2 = Middle, 3 = Lower)')
plt.ylabel('Count')
plt.savefig('assets/pclass_distribution.png')
plt.show()

# -----------------------------------------
# Plot 5: Heatmap of Correlation
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig('assets/correlation_heatmap.png')
plt.show()

# -----------------------------------------
# Plot 6: Survival Rate by Class and Gender
plt.figure(figsize=(8, 5))
sns.barplot(x='pclass', y='survived', hue='sex', data=df, palette='pastel')
plt.title('Survival Rate by Class and Gender')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.savefig('assets/survival_rate_by_class_gender.png')
plt.show()

# -----------------------------------------
# Plot 7: Embarkation Point vs Survival
plt.figure(figsize=(6, 4))
sns.countplot(x='embarked', hue='survived', data=df, palette='cool')
plt.title('Embarkation Point vs Survival')
plt.xlabel('Embarked Port (C = Cherbourg, Q = Queenstown, S = Southampton)')
plt.ylabel('Passenger Count')
plt.legend(title='Survived')
plt.savefig('assets/embarked_survival.png')
plt.show()

print("\n✅ All visualizations complete. Check the 'assets/' folder for saved images.")
