from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "SVM": SVC(kernel="linear", C=1.0, probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200, max_depth=6),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6)
    }
