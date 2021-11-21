import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.metadata.sortinghat import SortingHatFeatureMetadataEngine
from sklearn.model_selection import train_test_split

classif_data = {
    "NU": ["Cancer", "MFeat"],
    "CA": ["Nursery", "Audiology", "Hayes", "Supreme", "Flares", "Kropt", "Boxing"],
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

metadata = pd.read_csv("./metadata/metadata.csv")
sh_engine = SortingHatFeatureMetadataEngine()


def pipeline(data_dict, category, metadata):
    res = []
    for name in data_dict[category]:
        file_name = name.lower().replace(' ', '_') + '.csv'
        df = pd.read_csv('./data/' + file_name)
        truth_vec = metadata.loc[metadata.name == name].iloc[0, 3]
        label = list(df.columns)[-1]  # specifies which column do we want to predict
        data = TabularDataset(df)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=25)
        # perf0 = agl_downstream(df, train_data, test_data, label, predictor_type=0, truth_vec=truth_vec)
        perf0 = None
        perf1 = agl_downstream(df, train_data, test_data, label, predictor_type=1)
        perf2 = agl_downstream(df, train_data, test_data, label, predictor_type=2)
        res.append([name, perf0, perf1, perf2])
    return pd.DataFrame(res, columns=['Name', 'Truth', 'AGL', 'AGL+SH'])


'''
predictor_type: {0, 1, 2}
0 represents using true feature types
1 represents using AutoGluon auto-inferred feature types
2 represents using SortingHat inferred feature types
'''


def agl_downstream(df, train_data, test_data, label, predictor_type=1, truth_vec=None):
    # exclude other tree based models
    excluded = ['CAT', 'GBM', 'XT', 'custom']
    if predictor_type == 0:
        true_feature_metadata = sh_engine.to_feature_metadata(df, truth_vec)
        predictor = TabularPredictor(label=label, eval_metric="accuracy").fit(train_data,
                                                                              feature_metadata=true_feature_metadata,
                                                                              presets='best_quality',
                                                                              excluded_model_types=excluded)
    elif predictor_type == 2:
        predictor = TabularPredictor(label=label, eval_metric="accuracy").fit(train_data,
                                                                              use_metadata_engine=True,
                                                                              presets='best_quality',
                                                                              excluded_model_types=excluded)
    else:
        predictor = TabularPredictor(label=label, eval_metric="accuracy").fit(train_data,
                                                                              presets='best_quality',
                                                                              excluded_model_types=excluded)

    # results = predictor.fit_summary(show_plot=True)
    # Inference time:
    y_test = test_data[label]
    # delete labels from test data since we wouldn't have them in practice
    x_test = test_data.drop(labels=[label], axis=1)
    print(x_test.head())
    y_pred = predictor.predict(x_test)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    predictor.leaderboard(test_data)
    return perf


if __name__ == '__main__':
    perf_stats = pipeline(classif_data, 'CA+NG', metadata)
    print(perf_stats)
