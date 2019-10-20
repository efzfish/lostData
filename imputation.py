import numpy as np

class SmartImputation():
    def __init__(self, continuos_strategy, discrete_strategy, value=0, find_identical_cols=True, seed=2959):
        self.continuos_strategy = continuos_strategy
        self.discrete_strategy = discrete_strategy
        self.find_identical_cols = find_identical_cols
        self.value = value
        self.seed = seed

    @staticmethod
    def is_identical(col1, col2, max_nan_ratio=0.5):
        # TODO
        raise NotImplementedError

    def column_imputation(self, col, strategy, value=0):
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
        if strategy in ['mean', 'median', 'mode', 'constant', 'random']:
            self.column_imputation(col, strategy, value)
        else:
            # TODO
            raise NotImplementedError

    def discrete_imputation(self, col, matrix, strategy, value=0):
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
        

    
if __name__ == "__main__":
    pass