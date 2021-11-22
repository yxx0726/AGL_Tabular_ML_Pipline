import ast
import pandas as pd
from tabulate import tabulate
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.metadata.sortinghat import SortingHatFeatureMetadataEngine
from sklearn.model_selection import train_test_split

sh_engine = SortingHatFeatureMetadataEngine()

models_to_analyze = ["RandomForestGini_BAG_L1", "LightGBM_BAG_L1", "XGBoost_BAG_L1", "NeuralNetFastAI_BAG_L1"]


def pipeline(data_dict, category, metadata, downstream):
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
        bm0, perf0 = agl_downstream(name, df, train_data, test_data, label, downstream, predictor_type=0,
                                    truth_vec=truth_vec)
        bm1, perf1 = agl_downstream(name, df, train_data, test_data, label, downstream, predictor_type=1)
        bm2, perf2 = agl_downstream(name, df, train_data, test_data, label, downstream, predictor_type=2)
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


def agl_downstream(name, df, train_data, test_data, label, downstream, predictor_type=1, truth_vec=None):
    # exclude other tree based models
    print("Fitting downstream with predictor_type=%d" % predictor_type)
    excluded = [x for x in all_downstream_models if x != downstream]
    save_path = 'ag_models/' + name + '/' + downstream + '/'
    if predictor_type == 0:
        # truth
        true_feature_metadata = sh_engine.to_feature_metadata(df, truth_vec)
        predictor = TabularPredictor(label=label, eval_metric="accuracy", path=save_path).fit(train_data,
                                                                                              feature_metadata=true_feature_metadata,
                                                                                              presets='best_quality',
                                                                                              excluded_model_types=excluded)
    elif predictor_type == 2:
        # AG+SH
        predictor = TabularPredictor(label=label, eval_metric="accuracy", path=save_path).fit(train_data,
                                                                                              use_metadata_engine=True,
                                                                                              presets='best_quality',
                                                                                              excluded_model_types=excluded)
    else:
        # AG
        predictor = TabularPredictor(label=label, eval_metric="accuracy", path=save_path).fit(train_data,
                                                                                              presets='best_quality',
                                                                                              excluded_model_types=excluded)

    # results = predictor.fit_summary(show_plot=True)
    # Inference time:
    y_test = test_data[label]
    # delete labels from test data since we wouldn't have them in practice
    x_test = test_data.drop(labels=[label], axis=1)
    print(x_test.head())
    best_model_name = predictor.get_model_best()
    y_pred = predictor.predict(x_test, model=best_model_name)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred)
    return best_model_name, perf['accuracy']


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
    # pipeline(classif_data, 'CA', metadata, 'XGB')
    pipeline(classif_data, 'CA', metadata, 'FASTAI')
    # pipeline(classif_data, 'CA', metadata, 'RF')
