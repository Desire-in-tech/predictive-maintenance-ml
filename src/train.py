"""Train and export the Task 9 machine-failure model."""
import argparse, json
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_loader import load_ai4i, FEATURES, TARGET
from src.evaluation import choose_threshold, evaluate

def make_pipeline(model="random_forest"):
    numeric = [c for c in FEATURES if c != "Type"]
    prep = ColumnTransformer([
      ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
      ("type", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["Type"])
    ])
    estimators = {
      "random_forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
      "extra_trees": ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    }
    return Pipeline([("preprocessor", prep), ("classifier", estimators[model])])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model", choices=["random_forest","extra_trees"], default="random_forest")
    parser.add_argument("--min-recall", type=float, default=.50)
    args=parser.parse_args()
    df=load_ai4i(); X=df[FEATURES]; y=df[TARGET]
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)
    X_fit, X_val, y_fit, y_val=train_test_split(X_train,y_train,test_size=.20,stratify=y_train,random_state=42)
    pipe=make_pipeline(args.model).fit(X_fit,y_fit)
    threshold=choose_threshold(y_val,pipe.predict_proba(X_val)[:,1],args.min_recall)
    test_prob=pipe.predict_proba(X_test)[:,1]
    metrics=evaluate(y_test,test_prob,threshold)
    metrics.update({"model":args.model,"dataset":"UCI AI4I 2020","target":TARGET,
                    "features":FEATURES,"validation_recall_floor":args.min_recall})
    # Threshold is frozen before refitting on all non-test observations.
    pipe.fit(X_train,y_train)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    artifact={"pipeline":pipe,"threshold":threshold,"features":FEATURES,"target":TARGET,
              "model_name":args.model,"metrics":metrics,"dataset":"UCI AI4I 2020"}
    joblib.dump(artifact,out/"predictive_maintenance_pipeline.joblib")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    (out/"feature_schema.json").write_text(json.dumps({"features":FEATURES,"target":TARGET,"threshold":threshold},indent=2))
    print(json.dumps(metrics,indent=2))
    print(f"Artifact: {out/'predictive_maintenance_pipeline.joblib'}")

if __name__=="__main__": main()
