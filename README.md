# HR-Analytics-Project
This project is an HR analytics tool designed to predict employee turnover. By analyzing various employee data points such as satisfaction levels, performance evaluations, and salary, the project uses machine learning models to determine the likelihood of an employee leaving the company.
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

# Load the file HR.csv into pandas dataframe

# Read the file
df = pd.read_csv('C:/Users/baniy/Desktop/Projects/HR Analytics Project/HR.csv')

# Display the dataset
df.head()

# Display information about the dataset
df.info()

# Check the number of missing values in each column
df.isna().sum()

#Select only numerical values
num_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 
            'Work_accident', 'left', 'promotion_last_5years']
df_nums = df[num_cols]
df_nums.head()

# Calculate basic statistics for variables (mean, median, mode, min/max, standard deviation)

# Basic statistics of the dataset
df_nums.describe()

# Мode
df_nums.mode()

## Calculate and visualize the correlation matrix for quantitative variables. Identify the two most correlated and the two least correlated variables

# Create correlation matrices
correlation_matrix = df_nums.corr()
correlation_matrix

#Visualization
plt.subplots(figsize=(10, 10))
sns.heatmap(data=correlation_matrix, annot=True, cmap='Reds')

# The 2 most correlated variables
plt.subplots(figsize=(6, 6))
sns.heatmap(data=df_nums[['number_project', 'average_montly_hours']].corr(), annot=True, cmap='Blues')

# The 2 variables with the lowest correlation
plt.subplots(figsize=(6, 6))
sns.heatmap(data=df_nums[['time_spend_company', 'Work_accident']].corr(), annot=True, cmap='Blues')

### Calculate how many employees work in each department

# Count the number of occurrences of values in the department column 
df['department'].value_counts().sort_values()

# On the graph
df['department'].value_counts().plot(kind='bar')

# Show the distribution of employees by salary

# Count the number of occurrences of values in the 'salary' column.
df['salary'].value_counts()

df['salary'].value_counts().plot(kind='bar')

### Show the distribution of employees by salary in each department separately.

# Count the number of occurrences of values in the 'department' and 'salary' columns (considering only one column at a time).
df1 = df.groupby(['department', 'salary'])[['satisfaction_level']].count()
df1.columns = ['count']
df1

# Move one level of grouping from rows to columns and create a chart.
df1.unstack().plot(kind='barh', stacked=True, figsize=(10, 10))
plt.title("Salary by departments")
plt.xlabel("Count")
plt.ylabel("Department")

### Test the hypothesis that employees with higher salaries spend more time at work than employees with lower salaries.

# Extract the desired column into separate variables and filter by salary
df_high = df[df['salary'] == 'high']['average_montly_hours']
df_low = df[df['salary'] == 'low']['average_montly_hours']

# Hypothesis H0: Employees with high and low salaries work the same amount of time.
# Hypothesis H1: Employees with high salaries spend more time at work than employees with low salaries
# Calculate the t-statistic
t, p = stats.ttest_ind(a=df_high, b=df_low, equal_var=False)
t, p

#Usually, a significance level of 5%, or 0.05, is used
alpha = 0.05
if (p < alpha):
    print('Hypothesis H0 is rejected, and Hypothesis H1 is confirmed')
else:
    print('Hypothesis H0 is confirmed, and Hypothesis H1 is rejected')

# The same calculation of the t-statistic, but without using the stats library.
from math import sqrt

x1 = df_high.mean()
x2 = df_low.mean()
n1 = df_high.size
n2 = df_low.size
s1 = df_high.std()
s2 = df_low.std()

t = (x1 - x2) / sqrt(pow(s1, 2) / n1 + pow(s2, 2) / n2) 
t

# Calculate the following metrics for employees who have left and those who have not left (separately):

#### The proportion of employees who have received a promotion in the last 5 years

all_empl_count = df['promotion_last_5years'].count()

promotion_left_empl_count = df[df['left'] == 1]['promotion_last_5years'].sum()
promotion_empl_count = df[df['left'] == 0]['promotion_last_5years'].sum()

percentage_left = round(promotion_left_empl_count / all_empl_count * 100, 2)
percentage_not_left = round(promotion_empl_count / all_empl_count * 100, 2)

print(f'The proportion of employees who have left with a promotion in the last 5 years is {percentage_left}%')
print(f'The proportion of employees who have not left with a promotion in the last 5 years is {percentage_not_left}%')

#### Average level of satisfaction

satisfaction_left = round(df[df['left'] == 1]['satisfaction_level'].mean(), 2)
satisfaction_not_left = round(df[df['left'] == 0]['satisfaction_level'].mean(), 2)

print(f'The average level of satisfaction for employees who have left is {satisfaction_left}%')
print(f'The average level of satisfaction for employees who have not left is {satisfaction_not_left}%')

#### Average Number of Projects

projects_left = round(df[df['left'] == 1]['number_project'].mean(), 2)
projects_not_left = round(df[df['left'] == 0]['number_project'].mean(), 2)

print(f'The average number of projects for employees who have left is {projects_left}')
print(f'The average number of projects for employees who have not left is {projects_not_left}')

# Split the data into training and test sets. Build an LDA model to predict whether an employee has left based on available factors (excluding department and salary). Evaluate the model's performance on the test set.

# Extract the dependent variable
target = 'left'

# View the distribution of values
df[target].value_counts().plot(kind='bar')

# The values of the dependent variable are imbalanced (0 is significantly more than 1), so we specify the stratify parameter
# The values of the dependent variable are imbalanced (0 is significantly more than 1), so we specify the stratify parameter
Y = df_nums.pop(target)
X = df_nums.copy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
X_train.shape, X_test.shape

#Distribution of the dependent variable in the training set
y_train.value_counts().plot(kind='bar')

# Distribution of the dependent variable in the test set
y_test.value_counts().plot(kind='bar')

#Building the model
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# Checking the accuracy of predictions
print(f"Model accuracy on the training set: {100 * metrics.accuracy_score(y_train, pred_train)}%")
print(f"Model accuracy on the test set: {100 * metrics.accuracy_score(y_test, pred_test)}%")

# Checking the accuracy of predictions with a different metric (more appropriate)
print(f"Balanced accuracy of the model on the training set: {100 * metrics.balanced_accuracy_score(y_train, pred_train)}%")
print(f"Balanced accuracy of the model on the test set: {100 * metrics.balanced_accuracy_score(y_test, pred_test)}%")

# Creating the confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, pred_test)

sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
yticklabels=['Not Left', 'Left']
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.show()

# Model accuracy improved after including additional non-numeric parameters
df_new = pd.get_dummies(df, drop_first=True)
df_new.head()

#ebuilding the model with the new data
Y = df_new.pop(target)
X = df_new.copy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
X_train.shape, X_test.shape

#Building and evaluating the new model
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
print(f"Model accuracy on the training set: {100 * metrics.accuracy_score(y_train, pred_train)}%")
print(f"Model accuracy on the test set: {100 * metrics.accuracy_score(y_test, pred_test)}%")

print(f"Balanced accuracy of the model on the training set: {100 * metrics.balanced_accuracy_score(y_train, pred_train)}%")
print(f"Balanced accuracy of the model on the test set: {100 * metrics.balanced_accuracy_score(y_test, pred_test)}%")

#Creating the confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, pred_test)

sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
yticklabels=['Not Left', 'Left']
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.show()

# The accuracy of predictions has increased.
