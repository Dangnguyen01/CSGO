import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

data = pd.read_csv("dataset\csgo.csv")
data.drop(["team_a_rounds", "team_b_rounds", "date"], axis=1, inplace=True)
# data.info()

cat_dtype = ["map", "day", "month", "year", "result"]

def convert_cat(data, features):
    for feature in features:
        data[feature] = data[feature].astype("category")

convert_cat(data, cat_dtype)

x = data.drop("result", axis=1)
y = data["result"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

cat_features = ["map", "day", "month", "year"]
num_features = ["wait_time_s", "match_time_s", "ping", "kills", "assists", "deaths", "mvps"]

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.176, random_state=42)

# # Model
# log_reg_cv = LogisticRegression(solver="liblinear", max_iter=1000)
# dt_cv = DecisionTreeClassifier(criterion="entropy", max_depth=7, random_state=42)

# # Cross validation
# lr_score = cross_val_score(log_reg_cv, x_train, y_train, scoring="accuracy", cv=5)
# print(lr_score)
# print(lr_score.mean(), lr_score.std())

# print("--------------------------------------")
# dt_score = cross_val_score(dt_cv, x_train, y_train, scoring="accuracy", cv=5)
# print(dt_score)
# print(dt_score.mean(), dt_score.std())

# Baseline Model Comparison 
seed = 42
models = [
    LinearSVC(max_iter=12000, random_state=seed),
    SVC(random_state=seed),
    KNeighborsClassifier(metric="minkowski", p=2),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
    ExtraTreesClassifier(),
    AdaBoostClassifier(),
    # XGBClassifier(eval_metric="logloss", random_state=seed)
]

def generate_baseline_results(models, x, y, metrics, cv=5, plot_results=False):
    # define k-fold:
    kfold = StratifiedKFold(cv, shuffle=True, random_state=seed)
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        scores = cross_val_score(model, x, y, scoring=metrics, cv=kfold)
        for fold_idx, score in enumerate(scores):
            entries.append((model_name, fold_idx, score))
        
    cv_df = pd.DataFrame(entries, columns=["model_name", "fold_idx", "accuracy_score"])

    if plot_results:
        sns.boxplot(x="model_name", y="accuracy_score", data=cv_df, color="lightblue", showmeans=True)
        plt.title("Boxplot of Base-Line Model Accuracy using 5-fold cross-validation")
        plt.xticks(rotation=45)
        plt.show()

    # Summary result
    mean = cv_df.groupby("model_name")["accuracy_score"].mean()
    std = cv_df.groupby("model_name")["accuracy_score"].std()

    baseline_results = pd.concat([mean, std], axis=1, ignore_index=True)
    baseline_results.columns = ["Mean", "Standard Deviation"]

    # sort by accuracy
    baseline_results.sort_values(by=["Mean"], ascending=False, inplace=True)

    return baseline_results
    
print(generate_baseline_results(models, x_train, y_train, metrics="accuracy", plot_results=True))