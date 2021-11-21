from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

# Training time:
data = TabularDataset('data/hayes.csv')  #returns Pandas DataFrame
train_data, test_data = train_test_split(data, test_size=0.2, random_state=25)

#train_data = train_data.head(500)
print(train_data.head())
label = 'class'  # specifies which column do we want to predict
save_path = 'ag_models/'  # where to save trained models

# Parameters in the Tabluar Predictor:
# add metadata = True when using SH

# eval_metric Available for classification:
#[‘accuracy’, ‘balanced_accuracy’, ‘f1’, ‘f1_macro’, ‘f1_micro’, ‘f1_weighted’, 
#‘roc_auc’, ‘roc_auc_ovo_macro’, ‘average_precision’, ‘precision’, ‘precision_macro’,
#‘precision_micro’, ‘precision_weighted’, ‘recall’, ‘recall_macro’, ‘recall_micro’, 
#‘recall_weighted’, ‘log_loss’, ‘pac_score’]

# eval_metric Available for regression:
#[‘root_mean_squared_error’, ‘mean_squared_error’, ‘mean_absolute_error’, ‘median_absolute_error’, ‘r2’]

# AG
# predictor = TabularPredictor(label=label, eval_metric= "accuracy", path=save_path).fit(train_data, presets='best_quality',hyperparameters = {'NN':{}, 'XGB':{}})
# AG + SH
predictor = TabularPredictor(label=label, eval_metric= "accuracy", path=save_path).fit(train_data, use_metadata_engine=True, presets='best_quality',hyperparameters = {'NN':{}, 'XGB':{}})
results = predictor.fit_summary(show_plot=True)

# Inference time:
y_test = test_data[label]
test_data = test_data.drop(labels=[label], axis=1)  # delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = TabularPredictor.load(save_path)  # Unnecessary, we reload predictor just to demonstrate how to load previously-trained predictor from file
y_pred = predictor.predict(test_data)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
#predictor.leaderboard(test_data)
