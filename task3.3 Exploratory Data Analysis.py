import seaborn as sns
import matplotlib.pyplot as plt

# Churn distribution
sns.countplot(x="Churn_Yes", data=df)
plt.title("Churn Distribution")
plt.show()