import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline


def merge_files(medical_data, personal_data):
    personal_data = personal_data.drop(personal_data.columns[0], axis=1)
    medical_data = medical_data.drop(medical_data.columns[0], axis=1)
    data = pd.merge(personal_data, medical_data, on=["name", "address"], how="outer")
    return data


def rename_columns(X):
    X = X.rename(columns = {'fnlwgt': 'final_weight', 'class': 'diabetes_presence'}, inplace = False)
    return X


def unify_date_format(X):
    X["date_of_birth"] = X["date_of_birth"].map(format_date)
    return X


def format_date(date):
    date = str(date).replace("/", "-")
    date = date[:10]
    date = date.split("-")

    # if date format DD-MM-YYYY
    if len(date[2]) == 4:
        date = date[2] + "-" + date[1] + "-" + date[0]
        return date

    # if date format YY-MM-DD
    elif len(date[0]) == 2 and len(date[2]) == 2:
        date = "19" + date[0] + "-" + date[1] + "-" + date[2]
        return date

    date = "-".join(date)
    return date


def fix_age(X):
    age_median = X[(X["age"] > 0)].age.median()
    X.loc[(X["age"] < 0), "age"] = int(age_median)
    return X


def unify_sex_format(X):
    X['sex'] = X["sex"].map(fix_sex_value)
    return X


def fix_sex_value(sex):
    if sex.strip() == "Male":
        return 1
    else:
        return 0


def unify_pregnancy_format(X):
    X["pregnant"] = X["pregnant"].map(format_pregnancy)
    return X


def format_pregnancy(value):
    try:
        if value.strip() in ['t', 'T', 'TRUE']:
            return 1
        elif value.strip() in ['f', 'F', 'FALSE']:
            return 0
        else:
            return np.nan
    except AttributeError:
        return np.nan


def fix_male_pregnancy(X):
    X.loc[(X.sex == 1), "pregnant"] = 0
    return X


def delete_duplicates(X):
    X = X.drop_duplicates(["name", "address"])
    return X


column_dict = {}

class CleanData(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = unify_date_format(X)
        X = delete_duplicates(X)
        X = fix_male_pregnancy(X)
        X = unify_pregnancy_format(X)
        X = unify_sex_format(X)
        X = fix_age(X)
        X = rename_columns(X)

        X = X.drop(columns=['name', 'address', 'relationship', 'personal_info', 'education', 'income', 'date_of_birth'])

        list_columns = X.columns.tolist()
        for col in list_columns:
            column_dict[col] = list_columns.index(col)

        return X.values


# method = median alebo mean
class MissingValues(TransformerMixin):
    def __init__(self, method, columns):
        self.method = method
        self.columns = columns
        self.imputer = None

        if method == 'median':
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        if method == 'mean':
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:

            id = column_dict[col]

            if self.method == 'median':

                median = np.nanmedian(X[:, id])
                print("Median stlpca", col, median)

                for i, value in enumerate(X[:, id]):
                    if pd.isnull(value) or value == np.nan:
                        X[i][id] = median

            elif self.method == 'mean':
                mean = np.nanmean(X[:, id])

                for i, value in enumerate(X[:, id]):
                    if pd.isnull(value) or value == np.nan:
                        X[i][id] = mean

            elif self.method == 'delete':
                # X.drop(X[X[col].isnull()].index, inplace = True)
                indices_to_delete = []
                for i, value in enumerate(X[:, id]):
                    if pd.isnull(value) or value == np.nan:
                        indices_to_delete.append(i)

                X = np.delete(X, indices_to_delete, axis=0)

        return X


# Interquartile outlier removal
class OutlierDetection(TransformerMixin):
    def __init__(self, method, columns):
        self.method = method
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:

            id = column_dict[col]

            quantiles = np.quantile(X[:, id], [0.05, 0.95])

            for i, value in enumerate(X[:, id]):

                if value < quantiles[0]:
                    X[i][id] = np.nan if self.method == 'nan' else quantiles[0]
                if value > quantiles[1]:
                    X[i][id] = np.nan if self.method == 'nan' else quantiles[1]

            # print('Identified outliers: %d' % len(outliers))

        return X


class Imputer:
    def __init__(self, columns, neighbours):
        self.columns = columns
        self.imputer = KNNImputer(n_neighbors=neighbours)

    def fit(self, X, y=None):
        # for col in self.columns:
        #   X[col] = X[col].reshape([len(X[col]), 1])

        return self

    def transform(self, X):
        for col in self.columns:
            id = column_dict[col]

            lenght = len(X[:, id])
            reshaped_col = X[:, id].reshape(lenght, 1)
            self.transformer = self.imputer.fit(reshaped_col)

            X[:, id] = self.imputer.transform(reshaped_col).reshape(lenght)

        return X


class PowerTransform(TransformerMixin):
    def __init__(self, method, columns):
        self.columns = columns
        self.transformer = PowerTransformer(method=method, standardize=True)

    def fit(self, X, y=None):
        # for col in self.columns:
        #   X[col] = X[col].reshape([len(X[col]), 1])

        return self

    def transform(self, X):
        for col in self.columns:
            id = column_dict[col]

            lenght = len(X[:, id])
            reshaped_col = X[:, id].reshape(lenght, 1)
            self.transformer = self.transformer.fit(reshaped_col)

            X[:, id] = self.transformer.transform(reshaped_col).reshape(lenght)

        return X


list_delete = ['diabetes_presence']
list_mean = ['kurtosis_oxygen','skewness_glucose', 'mean_glucose', 'std_oxygen', 'skewness_oxygen', 'kurtosis_glucose', 'std_glucose', 'mean_oxygen', 'final_weight']
list_median = ['education-num', 'capital-gain', 'age', 'pregnant', 'hours-per-week', 'capital-loss']
list_outliers = [
    'kurtosis_oxygen', 'skewness_glucose', 'mean_glucose', 'std_oxygen',
    'skewness_oxygen', 'kurtosis_glucose', 'std_glucose', 'mean_oxygen',
    'capital-gain', 'final_weight', 'capital-loss'
]
list_transform_yeo = ['education-num', 'kurtosis_oxygen', 'skewness_glucose', 'mean_glucose', 'std_oxygen', 'skewness_oxygen', 'kurtosis_glucose', 'std_glucose', 'mean_oxygen']

pipeline1 =  Pipeline([
    ('Clean_data', CleanData()),
    ('Outliers', OutlierDetection('percentile', list_outliers)),
    ('Missing_values_delete',  MissingValues('delete', list_delete)),
    ('Missing_values_median',  MissingValues('median', list_median)),
    ('Missing_values_mean',  MissingValues('mean', list_mean)),
    # ('Outliers', OutlierDetection('percentile', list_outliers)),
    ('Transformer_yeo', PowerTransform('yeo-johnson', list_transform_yeo)),
])


list_delete2 = ['diabetes_presence']
list_mean2 = []
list_median2 = ['education-num', 'capital-gain', 'age', 'pregnant', 'hours-per-week', 'capital-loss']
list_outliers_nan2 = [
    'kurtosis_oxygen', 'skewness_glucose', 'mean_glucose', 'std_oxygen',
    'skewness_oxygen', 'kurtosis_glucose', 'std_glucose', 'mean_oxygen',
]

list_outliers_percentile2 = ['capital-gain', 'final_weight', 'capital-loss']
list_transform_yeo2 = ['education-num', 'kurtosis_oxygen', 'skewness_glucose', 'mean_glucose', 'std_oxygen', 'skewness_oxygen', 'kurtosis_glucose', 'std_glucose', 'mean_oxygen']
list_imputer2 = ['kurtosis_oxygen','skewness_glucose', 'mean_glucose', 'std_oxygen', 'skewness_oxygen', 'kurtosis_glucose', 'std_glucose', 'mean_oxygen', 'final_weight']

pipeline2 =  Pipeline([
    ('Clean_data', CleanData()),
    ('Missing_values_delete',  MissingValues('delete', list_delete2)),
    ('Missing_values_median',  MissingValues('median', list_median2)),
    ('Outliers_nan', OutlierDetection('nan', list_outliers_nan2)),
    ('Knn_impute', Imputer(list_imputer2, 10)),
    ('Outliers_percentile', OutlierDetection('percentile', list_outliers_percentile2)),
    ('Transformer_yeo', PowerTransform('yeo-johnson', list_transform_yeo2)),
])


def preprocess_dataset(medical_data, personal_data, flag):

    data = merge_files(medical_data, personal_data)
    data_pipe = data.copy(deep = True)
    
    
    if flag == 1:
        transformed = pipeline1.fit_transform(data_pipe)
    if flag == 2:
        transformed = pipeline2.fit_transform(data_pipe)

    all_columns = [k for k in column_dict]
    data_final = pd.DataFrame(data = transformed, columns = all_columns)

    return data_final
