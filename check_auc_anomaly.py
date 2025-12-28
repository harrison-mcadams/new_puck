import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv('data/20252026/20252026_df.csv')
df = df[df['event'].isin(['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot'])]
df['is_goal'] = (df['event'] == 'goal').astype(int)

# Standard features
features = ['distance', 'angle_deg', 'score_diff', 'period_number', 'time_elapsed_in_period_s', 'total_time_elapsed_s']

# Drop NaNs
df = df.dropna(subset=features + ['is_goal'])

# Split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Train a fast RF with no depth limit (like the script did for Standard)
clf = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=42)
clf.fit(df_train[features], df_train['is_goal'])

# Predict
p_train = clf.predict_proba(df_train[features])[:, 1]
p_test = clf.predict_proba(df_test[features])[:, 1]

auc_train = roc_auc_score(df_train['is_goal'], p_train)
auc_test = roc_auc_score(df_test['is_goal'], p_test)

print(f"Train AUC: {auc_train:.4f}")
print(f"Test AUC: {auc_test:.4f}")

# Importance
imps = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(imps)

# Check for duplicates across train/test
train_keys = set(zip(df_train['game_id'], df_train['total_time_elapsed_s'], df_train['distance'], df_train['angle_deg']))
test_keys = set(zip(df_test['game_id'], df_test['total_time_elapsed_s'], df_test['distance'], df_test['angle_deg']))

overlap = train_keys.intersection(test_keys)
print(f"\nExact key overlap between train and test: {len(overlap)}")
