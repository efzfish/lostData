import numpy as np
import sys

class SmartImputation():
    def __init__(self, continuos_strategy, discrete_strategy, value=0, find_identical_cols=True, seed=2959):
        self.continuos_strategy = continuos_strategy
        self.discrete_strategy = discrete_strategy
        self.find_identical_cols = find_identical_cols
        self.value = value
        self.seed = seed

    @staticmethod
    def is_identical(col1, col2, max_nan_ratio=0.5):
        # Identify if two columns have same pattern, so that missed data in one can be easily imputed
        # Example:
        # col1 = ['United States', 'Indonesia', NaN, 'Indonesia']
        # col2 = ['US', 'ID', 'ID','ID']
        # is_identical(col1, col2) returns True
        # And third element is col1 (NaN) can be filled as 'ID'
        assert len(col1) == len(col2), "Two columns have different number of rows"
        def helper(col1, col2):
            dic = {}
            for i, j in zip(col1, col2):
                if i and j:
                    dic.update({i: j})
            for j in range(len(col1)):
                for k, v in dic.items():
                    if (col1[j]==k and col2[j]!=v):
                        return False
            return True
        return helper(col1, col2) and helper(col2, col1)

    @staticmethod
    def get_missing_stats(df):
        # Returns missing stats of a dataframe
        return df.isna().sum() / len(df)

    def row_imputation(self, df, algo):
        # Do imputation based on another model based on the row (instance)
        # algo can be any machine learning method in scikit-learn or customized one
        # e.g. LinearRegressionRegressor, LogisticRegressionClassifier
        # e.g. KNearestNeighborClassifier, RandomForestClassifier, RandomForestRegressor
        df_nonan = df[~df.isnull().any(axis=1)]
        df_nan = df[df.isnull().any(axis=1)]
        column_nan = df.isna().any()
        algo.fit(df_nonan.loc[:,~column_nan], df_nonan.loc[:,column_nan])
        df[df.isnull().any(axis=1)] = algo.predict(df_nan.loc[:,~column_nan])
        return df

    def column_imputation(self, col, strategy, value=0):
        # Do imputation based on statistics/distribution over the column (feature)
        if strategy == 'mean':
            col[np.isnan(col)] = np.mean(col[~np.isnan(col)])
        elif strategy == 'median':
            col[np.isnan(col)] = np.mean(col[~np.isnan(col)])
        elif strategy == 'mode':
            # TODO
            raise NotImplementedError
        elif strategy == 'constant':
            col[np.isnan(col)] = value
        elif strategy == 'random':
            num_nan = sum(np.isnan(col))
            np.random.seed(self.seed)
            random_assignments = np.random.choice(col[~np.isnan(col)], num_nan)
            col[np.isnan(col)] = random_assignments

    def continuos_imputation(self, col, matrix, strategy, value=0):
        # Imputation on continuous variables
        if strategy in ['mean', 'median', 'mode', 'constant', 'random']:
            self.column_imputation(col, strategy, value)
        else:
            # TODO
            raise NotImplementedError

    def discrete_imputation(self, col, matrix, strategy, value=0):
        # Imputation on discrete variables
        if strategy in ['mode', 'random']:
            self.column_imputation(col, strategy, value)
        else:
            # TODO
            raise NotImplementedError

    def impute(self, df):
        cols = list(df.columns())
        if self.find_identical_cols:
            for idx1, col1 in enumerate(cols):
                for _, col2 in enumerate(cols[idx1+1:]):
                    if self.is_identical(df[col1], df[col2]):
                        df[col2] = df[col1]
        # TODO
        return df
        

    
if __name__ == "__main__":
    with open(sys.argv[1]) as fin, open(sys.argv[2], 'w') as fout:
        si = SmartImputation(sys.argv[3], sys.argv[4])
        imputed = si.impute(fin)
        imputed.to_csv(fout)
