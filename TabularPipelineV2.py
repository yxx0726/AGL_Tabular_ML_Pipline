#Copyright 2022 Jin Dai, Xin Yu, Arun Kumar
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import ast
import pandas as pd
from tabulate import tabulate
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.metadata.sortinghat import SortingHatFeatureMetadataEngine
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None, 'display.max_colwidth', None)

sh_engine = SortingHatFeatureMetadataEngine()


'''
pipeline takes 5 parameters:
- data_dict: classif_data or reg_data; type=dict
- category: the feature types e.g. 'CA' or 'NU+CA'; type=str
- metadata: the metadata df; type=pd.DataFrame
- downstream: the downstream model type e.g. 'XGB', 'FASTAI' or 'RF'; type=str
- problem_type: the problem type, whether it is 'regression' or 'classification'
'''


def pipeline(data_dict, category, metadata, downstream, problem_type=None):
    res = []
    formatter = "{model}|{perf:.4f}"
    for name in data_dict[category]:
        print("Fitting dataset: %s" % name)
        file_name = name.lower().replace(' ', '_') + '.csv'
        df = pd.read_csv('./data/' + file_name)
        truth_vec = ast.literal_eval(metadata.loc[metadata.name == name].iloc[0, 3])
        label = metadata.loc[metadata.name == name].iloc[0, 2]  # specifies which column do we want to predict
        data = TabularDataset(df)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=25)

        bm0, perf0 = agl_downstream(name, df, train_data, test_data, label, downstream,
                                    problem_type=problem_type, predictor_type=0, truth_vec=truth_vec)
        bm1, perf1 = agl_downstream(name, df, train_data, test_data, label, downstream,
                                    problem_type=problem_type, predictor_type=1)
        bm2, perf2 = agl_downstream(name, df, train_data, test_data, label, downstream,
                                    problem_type=problem_type, predictor_type=2)
        res.append(
            [name, downstream,
             formatter.format(model=bm0, perf=perf0),
             formatter.format(model=bm1, perf=perf1),
             formatter.format(model=bm2, perf=perf2)]
        )
    print(tabulate(res, headers=['Name', 'Downstream', 'Truth', 'AGL', 'AGL+SH']))


'''
predictor_type: {0, 1, 2}
0 represents using true feature types
1 represents using AutoGluon auto-inferred feature types
2 represents using SortingHat inferred feature types
'''
'''
Stable downstream model types:
'GBM' (LightGBM)
'CAT' (CatBoost)
'XGB' (XGBoost)
'RF' (random forest)
'XT' (extremely randomized trees)
'KNN' (k-nearest neighbors)
'LR' (linear regression)
'NN' (neural network with MXNet backend)
'FASTAI' (neural network with FastAI backend)
'''
all_downstream_models = ['GBM', 'CAT', 'XGB', 'RF', 'XT', 'LR', 'FASTAI', 'NN', 'KNN']


def agl_downstream(name, df, train_data, test_data, label, downstream,
                   problem_type=None, predictor_type=1, truth_vec=None):
    # exclude other tree based models
    print("Fitting downstream with predictor_type=%d" % predictor_type)
    excluded = [x for x in all_downstream_models if x != downstream]
    save_path = 'ag_models/' + name + '/' + downstream + '/'
    eval_metric = None
    if problem_type == 'regression':
        eval_metric = 'root_mean_squared_error'
    elif problem_type == 'classification':
        eval_metric = 'accuracy'
        problem_type = None     # let AGL infer the classification type - binary or multiclass
    if predictor_type == 0:
        # truth
        true_feature_metadata = sh_engine.to_feature_metadata(df, truth_vec)
        predictor = TabularPredictor(label=label, eval_metric=eval_metric, problem_type=problem_type, path=save_path)\
            .fit(train_data,
                 feature_metadata=true_feature_metadata,
                 presets='best_quality',
                 num_bag_folds=0,
                 num_stack_levels=0,
                 excluded_model_types=excluded)
    elif predictor_type == 2:
        # AG+SH
        predictor = TabularPredictor(label=label, eval_metric=eval_metric, problem_type=problem_type, path=save_path)\
            .fit(train_data,
                 use_metadata_engine=True,
                 presets='best_quality',
                 num_bag_folds=0,
                 num_stack_levels=0,
                 excluded_model_types=excluded)
    else:
        # AG
        predictor = TabularPredictor(label=label, eval_metric=eval_metric, problem_type=problem_type, path=save_path)\
            .fit(train_data,
                 presets='best_quality',
                 num_bag_folds=0,
                 num_stack_levels=0,
                 excluded_model_types=excluded)
    # Inference time:
    y_test = test_data[label]
    # delete labels from test data since we wouldn't have them in practice
    x_test = test_data.drop(labels=[label], axis=1)
    best_model_name = predictor.get_model_best()
    y_pred = predictor.predict(x_test, model=best_model_name)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred)
    lb = predictor.leaderboard(test_data, silent=True, extra_info=True)
    print(lb[lb.model == best_model_name]['hyperparameters'])
    print(lb[lb.model == best_model_name]['hyperparameters_fit'])
    print(lb[lb.model == best_model_name]['child_hyperparameters'])
    print(lb[lb.model == best_model_name]['child_hyperparameters_fit'])
    return best_model_name, perf[eval_metric]


classif_data = {
    "NU": ["Cancer", "Mfeat"],
    # "CA": ["Nursery", "Audiology", "Hayes", "Supreme", "Flares", "Kropt", "Boxing"],
    "CA": ["Nursery", "Hayes", "Supreme", "Flares", "Kropt", "Boxing"],
    "CA+NG": ["Apnea2"],
    "NU+CA": ["Flags", "Diggle", "Hearts", "Sleuth"],
    "NU+CA+ST": ["Auto-MPG"],
    "NU+CA+ST+NG": ["Clothing"],
    "NU+DT+NG": ["IOT"],
    "NU+DT+EN": ["NYC"],
    "ST": ["BBC"],
    "DT+ST": ["Articles"],
    "NG+CA": ["Zoo"],
    "NU+CA+EN": ["Churn"],
    "NU+CA+EN+NG": ["PBCseq"],
    "NU+CA+LST+NG+CS": ["Pokemon"],
    "NU+CA+DT+URL+NG+CS": ["President"]
}

reg_data = {
    "CA": ["MBA"],
    "NU+CA": ["Vineyard", "Apnea"],
    "DT": ["Accident"],
    "NU+CA+EN+NG": ["Car Fuel"]
}

if __name__ == '__main__':
    metadata = pd.read_csv("./metadata/metadata.csv")
    # sample run
    pipeline(reg_data, 'CA', metadata, 'RF', 'regression')
    # uncomment and modify parameters of the below lines for other classification/regression tasks
    # and switching downstream models
    # pipeline(classif_data, 'NU+CA+EN+NG', metadata, 'XGB', 'classification')
    # pipeline(classif_data, 'NU+CA+EN+NG', metadata, 'FASTAI', 'classification')
    # pipeline(classif_data, 'NU+CA', metadata, 'RF', 'classification')
    # pipeline(reg_data, 'CA', metadata, 'RF', 'regression')
    # pipeline(reg_data, 'CA', metadata, 'XGB', 'regression')
    # pipeline(reg_data, 'CA', metadata, 'FASTAI', 'regression')
