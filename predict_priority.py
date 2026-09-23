import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def print_section(title):
    print("\n" + "=" * 70)
    print(title.upper())
    print("=" * 70)

# Load the data set

data = pd.read_csv('Support_tickets.csv')

# Create a copy
data_copy = data.copy()


# Check column types and look for missing values
# print(data_copy.info())

print_section("Data Cleaning")
# Check shape before and after dropping duplicates to confirm if any existed

print(f"Shape before dropping duplicates: {data_copy.shape}")
data_copy = data_copy.drop_duplicates()
print(f"Shape after dropping duplicates: {data_copy.shape}")


print_section("Customer Sentiment Investigation")
# Check whether customer_sentiment has a meaningful effect on priority
print(data_copy.groupby('priority_cat')['customer_sentiment'].value_counts())
print(data_copy.groupby('priority_cat')['customer_sentiment_cat'].mean())


# Drop raw text columns since numeric encodings already exist for them.
# Also drop customer_sentiment and customer_sentiment_cat since the relationship to priority was too weak to be useful.
data_copy = data_copy.drop(columns = [
    "ticket_id",
    "company_id",
    "day_of_week",
    "company_size",
    "industry",
    "customer_tier",
    "region",
    "product_area",
    "booking_channel",
    "reported_by_role",
    "customer_sentiment",
    "customer_sentiment_cat",
    "priority"
]
)

# Save a sample of the cleaned (pre-scaling) data for reference
sampli_data_after_clean_up_1 = data_copy.head(100)
# Create the output directory if it doesn't exist
os.makedirs("output", exist_ok=True)
sampli_data_after_clean_up_1.to_csv('output/sampli_data_after_clean_up_1.csv', index=False)

# Scale only the truly continuous numeric columns.
# _cat columns and flag columns are encoded categories, not real quantities, so they are left unscaled.

Scaler = StandardScaler()
continuous_columns = [
    "org_users",
    "past_30d_tickets",
    "customers_affected",
    "past_90d_incidents",
    "error_rate_pct",
    "downtime_min",
    "description_length"
]
data_copy[continuous_columns] = Scaler.fit_transform(data_copy[continuous_columns])


# Save a sample of the final scaled data for reference
sampli_data_after_clean_up_2 = data_copy.head(100)
sampli_data_after_clean_up_2.to_csv('output/sampli_data_after_clean_up_2.csv', index=False)

#Save The Preprocessed Dataset as a csv
data_copy.to_csv('preprocessed_dataset.csv', index=False)


target = "priority_cat"

x = data_copy.drop(columns = [target])

y = data_copy[target]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)

model = RandomForestClassifier(random_state=10)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)


accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average="weighted", zero_division=0)
recall = recall_score(y_test, y_predict, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_predict, average="weighted", zero_division=0)
print_section("Baseline RandomForestClassifier Results")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


feature_importances = model.feature_importances_
feature_names = x.columns


sorted_idx = feature_importances.argsort()[::-1]

import matplotlib.pyplot as plt





# Plot feature importance to see which inputs the model relies on most
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_idx])
plt.xticks(range(len(feature_importances)), feature_names[sorted_idx], rotation=90)
plt.title("Feature Importance - RandomForestClassifier")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("output/feature_importance.png")


# Visualize downtime_min & customers_affected spread across priority levels
data_copy.boxplot(column='customers_affected', by='priority_cat')
plt.savefig("output/boxplot_customers_affected.png")

data_copy.boxplot(column='downtime_min', by='priority_cat')
plt.savefig("output/boxplot_downtime_min.png")


# Model optimization

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, x, y, cv=5, scoring = 'accuracy')
print_section("Cross-Validation Results")
print(f"Cross-Validation Scores: {cv_scores} | Mean: {cv_scores.mean()}")


# Model hyperparameter tuning using GridSearchCV

from sklearn.model_selection import GridSearchCV
import time

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}
start_time = time.time()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

end_time = time.time()
execution_time = end_time - start_time

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print_section("GridSearchCV Hyperparameter Tuning")
print(f"Best parameters: {best_params} \nBest score: {best_score}")
print(f"Execution time: {execution_time} seconds")
