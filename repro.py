import pandas as pd
import xgboost as xgb
import pdb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


def train():
	df_train = pd.read_csv("train_xgb.csv")
	df_valid = pd.read_csv("valid_xgb.csv")
	assert(df_train.shape[1] == df_valid.shape[1])

	features = list(filter(lambda x: "f_" in x, df_train))
	predictor = "predictor"
	x_train = df_train[features].values
	y_train = df_train[predictor].values

	x_valid = df_valid[features].values
	y_valid = df_valid[predictor].values

	trainer_params = {'learning_rate': '0.040', 'n_estimators': 110, 'max_depth': 7, 'colsample_bytree': '0.700', 'subsample': '0.700', 'min_child_weight': '3.000', 'gamma': '3.000', 'reg_lambda': '10.000', 'reg_alpha': '4.000', 'tree_method': 'hist'}

	fit_params={'early_stopping_rounds': 2, 
	                    'eval_metric': "logloss",
	                    'verbose': 2,
	                    'eval_set': [[x_valid, y_valid]]}

	clf = xgb.XGBClassifier(nthread = 1,use_label_encoder=False,**trainer_params)
	cross_val_scoring = "neg_log_loss"

	# This fails
	score = cross_val_score(clf, x_train, y_train, cv = 2, verbose = 3, scoring = cross_val_scoring, fit_params = fit_params)
	print(score)

	# This works
	score = cross_val_score(clf, x_valid, y_valid, cv = 2, verbose = 3, scoring = cross_val_scoring, fit_params = fit_params)
	print(score)

train()