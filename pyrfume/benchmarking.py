'''Pyrfume benchmarking module.'''

import warnings
import itertools
from typing import List, Union, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier
import pyrfume


RANDOM_STATE = 1984

ESTIMATOR_MAP = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'BernoulliNB': BernoulliNB(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'MultinomialNB': MultinomialNB(),
    'SVC': SVC(),
    'XGBClassifier': XGBClassifier(),
    'MLPClassifier': MLPClassifier(),
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'GaussianProcessRegressor': GaussianProcessRegressor(),
    'PLSRegression': PLSRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'MLPRegressor': MLPRegressor()
 }

DEFAULT_PARAMETERS_CLASSIFIERS = {
    'LogisticRegression': {
        'max_iter': [500],
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
        'dual': [False],
        'random_state': [RANDOM_STATE]
    },
    'DecisionTreeClassifier': {
        'min_impurity_decrease': np.arange(0, 0.005, 0.00025),
        'max_features': [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'random_state': [RANDOM_STATE]
    },
    'RandomForestClassifier': {
        'n_estimators': [10, 50, 100, 500],
        'min_impurity_decrease': np.arange(0, 0.005, 0.00025),
        'max_features': [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'random_state': [RANDOM_STATE]
    },
    'AdaBoostClassifier': {
        'learning_rate': [0.01, 0.1, 0.5, 1, 10, 50, 100],
        'n_estimators': [10, 50, 100, 500],
        'random_state': [RANDOM_STATE]
    },
    'BernoulliNB': {
        'alpha': [0, 0.1, 0.25, 0.5, 0.75, 1, 5, 10, 25, 50],
        'fit_prior': [True, False],
        'binarize': [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    },
    'ExtraTreesClassifier': {
        'n_estimators': [10, 50, 100, 500],
        'min_impurity_decrease': np.arange(0., 0.005, 0.00025),
        'max_features': [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
        'criterion':  ['gini', 'entropy', 'log_loss'],
        'random_state': [RANDOM_STATE]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [10, 50, 100, 500],
        'min_impurity_decrease': np.arange(0., 0.005, 0.00025),
        'max_features': [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
        'learning_rate': [0.01, 0.1, 0.5, 1, 10, 50, 100],
        'loss': ['log_loss'],
        'random_state': [RANDOM_STATE]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [1, 2, 3, 4, 5, 10, 20, 25, 30, 50, 100],
        'weights': ['uniform', 'distance']
    },
    'MultinomialNB': {
        'alpha': [0, 0.1, 0.25, 0.5, 0.75, 1, 5, 10, 25, 50],
        'fit_prior': [True, False]
    },
    'SVC': {
        'C': [0.01, 0.1, 0.5, 1, 10, 50, 100],
        'gamma': [0.01, 0.1, 0.5, 1, 10, 50, 100, 'auto'],
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'degree': [2, 3],
        'coef0': [0, 0.1, 0.5, 1, 10, 50, 100],
        'probability': [True],
        'random_state': [RANDOM_STATE]
    },
    'XGBClassifier': {
        'n_estimators': [10, 50, 100, 500],
        'learning_rate': [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
        'gamma': np.arange(0, 0.51, 0.05),
        'max_depth': [1, 2, 3, 4, 5, 10, 20, 50, None],
        'subsample': np.arange(0, 1.01, 0.1),
        'seed': [RANDOM_STATE]
    },
    'MLPClassifier': {
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'early_stopping': [True],
        'beta_1': np.arange(0.8, 0.99, 0.01),
        'beta_2': np.arange(0.99, 0.999, 0.001),
        'random_state': [RANDOM_STATE]
    }
}

DEFAULT_PARAMETERS_REGRESSORS = {
    'LinearRegression': {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },
    'Ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'solver': ['auto'],
        'normalize': [True, False],
        'random_state': [RANDOM_STATE]
    },
    'Lasso': {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'normalize': [True, False],
        'max_iter': [1000, 3000, 5000, 7000],
        'random_state': [RANDOM_STATE]
    },
    'ElasticNet': {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.3, 0.5, 0.8],
        'normalize': [True, False],
        'max_iter': [1000, 3000, 5000, 7000],
        'random_state': [RANDOM_STATE]
    },
    'SVR': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 5, 10, 50, 100],
        'epsilon': [0.01, 0.03, 0.05, 0.1, 0.2]
    },
    'KNeighborsRegressor': {
        'n_neighbors': [1, 4, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [5, 15, 30, 60, 100, 150]
    },
    'GaussianProcessRegressor': {
        'alpha': [1e-12, 1e-10, 1e-6, .001, 0.01, 0.1, 1],
        'random_state': [RANDOM_STATE]
    },
    'PLSRegression': {
        'max_iter': [100, 500, 1000],
        'tol': [1e-8, 1e-6, 1e-3]
    },
    'DecisionTreeRegressor': {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'min_samples_split': [1, 2, 5, 10],
        'random_state': [RANDOM_STATE]
    },
    'MLPRegressor': {
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [1e-5, 1e-4, 1e-3],
        'learning_rate': ['constant', 'adaptive'],
        'random_state': [RANDOM_STATE]
    }
}

SCORE_FUNC_DICT = {
    'classification' : {
        'f-statistic': f_classif,
        'mutual_info': mutual_info_classif,
        'chi2': chi2
    },
    'regression': {
        'f-statistic': f_regression,
        'mutual_info': mutual_info_regression
    }
}


class DatasetError(Exception):
    '''Exception to catch invalid datasets.'''


class PyrfumeDataset:
    '''Class to streamline Pyrfume data into ML pipelines.'''
    def __init__(self, archive: str, df: pd.DataFrame, task: str, feature_set: str = None):
        self.archive = archive
        self.target_name = df.columns[0]
        self.feature_set = feature_set
        self.task = task
        self.label_encoder = None

        if task == 'classification':
            # GridSearchCV requires n_splits >= 2
            self.n_splits = max(2, df[self.target_name].value_counts().min() // 2)
        else:
            self.n_splits = 5

        if feature_set is not None:
            # Add molecule features
            df = df.join(get_molecule_features(index=df.index, feature_set=feature_set))
            # Drop any molecules with no returned features
            df = df[~df.isnull().any(axis=1)]
        self.df = df

    def describe(self):
        '''Print summary of dataset.'''
        print('\n'.join([
            f'{"-" * 50}\nArchive: {self.archive}',
            f'Prediction target: {self.target_name}',
            f'# molecules: {self.df.shape[0]}',
            f'Molecule features: {self.feature_set}',
            f'# features: {self.df.shape[1] - 1}',
            f'ML task: {self.task}',
            f'n_splits: {self.n_splits}',
            f'\n{self.df.iloc[:, :3].head()}\n{"-" * 50}'
        ]))

    def add_features(self, feature_set: str, features: pd.DataFrame = None):
        '''Can be used to add features from scratch or append to existing feature set.'''
        if features is not None:
            self.df = self.df.join(features)
        else:
            self.df = self.df.join(
                get_molecule_features(index=self.df.index, feature_set=feature_set)
            )
        # Drop any molecules with no returned features
        self.df = self.df[~self.df.isnull().any(axis=1)]

        if self.feature_set is None:
            self.feature_set = feature_set
        else:
            self.feature_set += f'_{feature_set}'

    def get_features_targets(self):
        '''Return features, prediction targets.'''
        return self.df.drop(self.target_name, axis=1).values, self.df[self.target_name].values

    def get_feature_names(self):
        '''Return feature names.'''
        return self.df.drop(self.target_name, axis=1).columns.to_list()

    def get_cids(self):
        '''Return CIDs in dataset.'''
        return self.df.index.to_list()

    def encode_labels(self):
        '''Encode labels.'''
        self.label_encoder = LabelEncoder()
        self.df[self.target_name] = self.label_encoder.fit_transform(self.df[self.target_name])

    def plot_target_distribution(self):
        '''Plot distribution of prediction target values.'''
        if self.task == 'classification':
            # Plot frequency of label use
            counts = self.df[self.target_name].value_counts()
            counts.plot.bar(figsize=(12,4), fontsize=8, logy=True)
        if self.task == 'regression':
            n_bins = int(np.round(1 + 3.322 * np.log10(self.df.shape[0]))) # Sturge's Rule
            self.df.hist(column=self.target_name, bins=n_bins, figsize=(12,4))
        plt.ylabel('Frequency')
        plt.show()

    def threshold_labels(self, min_counts: int):
        '''Option to drop labels used at low freqeuncy.'''
        counts = self.df[self.target_name].value_counts()
        self.df = self.df[self.df[self.target_name].isin(counts[counts >= min_counts].index)]
        self.n_splits = min_counts // 2

    def set_n_splits(self, n_splits):
        '''Option to manually set # folds for cross-validation.'''
        self.n_splits = n_splits

    def plot_feature_correlations(self, method: str = 'spearman'):
        '''Plot correlations between molecule features.'''
        sns.set(font_scale = 0.8)
        plt.figure(figsize=(8,8))
        sns.heatmap(
            self.df.drop(columns=self.target_name).corr(method=method),
            cmap='bwr',
            cbar_kws={'label': f'{method} correlation'}
        )
        plt.show()

    def select_features(self, score_function: str, mode: str, param: Union[int, float, str]):
        '''Option to use Feature Selection transformation.'''
        selector = resolve_feature_selection(
            task=self.task,
            score_function=score_function,
            mode=mode
        )
        selector.set_params(**{'param': param})
        selector.fit(self.df.drop(columns=self.target_name), self.df[self.target_name])
        cols = [self.target_name] + list(selector.get_feature_names_out())
        self.df = self.df[cols]


class Model:
    '''Class to streamline batch pipeline processing.'''
    def __init__(
        self,
        estimator: Union[str, List[Callable]],
        param_grid: dict = None,
        scoring: Union[str, List[str]] = None,
        task: str = None
    ):
        # Using pre-configured default estimator
        if isinstance(estimator, str):
            self.steps = [ESTIMATOR_MAP[estimator]]
            estimator_name = type(ESTIMATOR_MAP[estimator]).__name__.lower()
            # These require MinMax scaling
            if estimator in ['BernoulliNB', 'MultinomialNB']:
                self.steps.insert(0, MinMaxScaler())
            if param_grid is None: # Use default values above
                self.param_grid = {estimator_name: get_default_parameters(estimator)}
            else:
                self.param_grid = {estimator_name: param_grid}
            if task is None:
                self.task = resolve_task(estimator)
            else:
                self.task = task
            self.scoring = resolve_scoring(scoring=scoring, task=self.task)
        # Using non-default estimator
        else:
            self.steps = estimator
            estimator_name = type(estimator).__name__.lower()
            if param_grid is None:
                raise ValueError('Must provide parameter grid.')
            self.param_grid = {estimator_name: param_grid}
            if scoring is None:
                raise ValueError('Must provide scoring metric(s).')
            self.scoring = scoring

        self.pipeline_string = pipeline_steps_to_string(self.steps)

    def describe(self):
        '''Prints model attributes.'''
        print('\n'.join([
            f'Pipeline: {self.pipeline_string}',
            f'Parameter grid: {self.param_grid}',
            f'Scoring: {self.scoring}',
            f'Task: {self.task}\n'
            ])
        )

    def add_step(self, step: Callable, position: int = 0, param_grid: dict = None):
        '''Add step to pipeline.'''
        self.steps.insert(position, step)
        self.pipeline_string = pipeline_steps_to_string(self.steps)
        if param_grid is not None:
            self.param_grid[type(step).__name__.lower()] = param_grid

    def add_feature_selection(self, score_function: str, mode: str, param_list: List):
        '''Add feature selection transformation for GridSearchCV.'''
        selector = resolve_feature_selection(
            task=self.task,
            score_function=score_function,
            mode=mode
        )
        # Add feature selector after MinMaxScaler if already part of pipeline
        if isinstance(self.steps[0], MinMaxScaler):
            position = 1
        else:
            position = 0
        self.add_step(step=selector, position=position, param_grid={'param': param_list})

    def set_parameter_grid(self, param_grid):
        '''Option to set custom parameter grid.
        
        Args:
            param_grid: When using this method, param_grid must be provided as a nested dict
                of the following format: {pipeline_step_name: {param_name: param_value}}.
        '''
        self.param_grid = param_grid


def resolve_feature_selection(
        task: str,
        score_function: str,
        mode: str
) -> GenericUnivariateSelect:
    '''Generator feature selection transformation.'''
    if mode not in ['percentile', 'k_best', 'fpr', 'fdr', 'fwe']:
        raise ValueError(f'Invalid selection mode {mode}')
    try:
        score_func = SCORE_FUNC_DICT[task][score_function]
    except KeyError as exc:
        raise KeyError(exc) from exc
    return GenericUnivariateSelect(score_func=score_func, mode=mode)


def resolve_task(estimator: str) -> str:
    '''Determine whether classification or regression task.'''
    if estimator in DEFAULT_PARAMETERS_CLASSIFIERS:
        return 'classification'
    if estimator in DEFAULT_PARAMETERS_REGRESSORS:
        return 'regression'
    raise ValueError(f'No defaults for {estimator}')


def get_default_parameters(estimator: str = None) -> dict:
    '''Return default parameter grid for specified estimator.'''
    if estimator is None:
        return {**DEFAULT_PARAMETERS_CLASSIFIERS, **DEFAULT_PARAMETERS_REGRESSORS}
    if estimator in DEFAULT_PARAMETERS_CLASSIFIERS:
        return DEFAULT_PARAMETERS_CLASSIFIERS[estimator]
    if estimator in DEFAULT_PARAMETERS_REGRESSORS:
        return DEFAULT_PARAMETERS_REGRESSORS[estimator]
    raise ValueError(f'No defaults for {estimator}')


def resolve_scoring(scoring: Union[str, List[str]], task: str = None) -> Union[str, List[str]]:
    '''Return scoring metrics for GridSearchCV.

    Args:
        scoring: Scoring metric(s) for GridSearchCV. If None, uses 'task' to return defaults.
        task: Specify 'classification' or 'regression'.

    Returns:
        Either single or list of scoring metric(s).

    Raises:
        ValueError: If scoring is None, and task is not 'classification' or 'regression'.
    '''
    if (scoring is None) & (task is None):
        raise ValueError('Must specify scoring method(s) or task.')
    if scoring is not None:
        return scoring
    if task == 'classification':
        return ['balanced_accuracy', 'accuracy', 'f1_macro', 'f1_weighted', 'roc_auc_ovr_weighted']
    if task == 'regression':
        return ['neg_root_mean_squared_error', 'max_error', 'r2']
    raise ValueError(f'Invalid task: {task}')


def pipeline_steps_to_string(pipeline_steps: List) -> str:
    '''Create string from steps in pipeline.'''
    return ';'.join([str(step).split('(', maxsplit=1)[0] for step in pipeline_steps])


def list_default_estimators(task: str = None) -> List:
    '''Return list of estimators that are provided with default parameter grids.'''
    if task is None:
        return list(DEFAULT_PARAMETERS_CLASSIFIERS.keys()) + \
            list(DEFAULT_PARAMETERS_REGRESSORS.keys())
    if task == 'classification':
        return list(DEFAULT_PARAMETERS_CLASSIFIERS.keys())
    if task == 'regression':
        return list(DEFAULT_PARAMETERS_REGRESSORS.keys())

    raise ValueError(f'Invalid task: {task}')


def get_molecule_features(index: pd.Index, feature_set: str = 'mordred') -> pd.DataFrame:
    '''Fetches specified features for molecule CIDs in 'index'.

    Mordred and Morgan features have already been scaled (StandardScaler) and imputed
        (KNNImputer) at the time of creation in the Pyrfume-Data repository. Here during retrieval,
        features will zero variance will be dropped.

    Args:
        index: molecule CIDs for which to fetch features.
        feature_set: Use 'mordred' for Mordred features, 'morgan' for Morgan similarity, or 
            'mordred_morgan' to merge these two feature sets.

    Returns:
        DataFrame of the specified feature set, indexed on CID.
    
    Raises:
        ValueError: if feature_set is not 'morgan', 'mordred', or 'mordred_morgan'.
    '''
    print('Loading features...')

    if feature_set == 'mordred':
        features = pyrfume.load_data('mordred/features.csv')
    elif feature_set == 'morgan':
        features = pyrfume.load_data('morgan/features_sim.csv')
    elif feature_set == 'mordred_morgan':
        features = pyrfume.load_data('mordred/features.csv')
        features = features.join(pyrfume.load_data('morgan/features_sim.csv'))
    else:
        raise ValueError(f'Could not find features for {feature_set}')

    # Common index between behavior dataframe and features dataframe
    common_index = index.intersection(features.index)
    features = features.loc[common_index]

    # Only keep features with non-zero variance
    features = features[features.columns[features.nunique() > 1]]

    print(f'Returned {features.shape[1]} features for {features.shape[0]} molecules')

    return features


def resolve_cv(task: str = None, n_splits: int = 5) -> Union[KFold, StratifiedKFold]:
    '''Return cross-validation generator for classification or regresson task.'''
    if task == 'classification':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    if task == 'regression':
        return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    raise ValueError(f'Invalid task: {task}')


def reformat_param_grid(parameters: dict) -> dict:
    '''Reformat parameter grid for GridSearchCV expecting Pipeline.

    Reformats Model.param_gird from dict of dicts to flat dict with convention:
    {step_name__param_name: param_values}

    Args:
        parameters: dict of parameter grid for pipeline steps.
    '''
    param_grid = {}
    for step_name, grid in parameters.items():
        for param_name, param_vals in grid.items():
            param_grid[f'{step_name}__{param_name}'] = param_vals
    return param_grid


def evaluate_model(
    dataset: PyrfumeDataset,
    pipeline: Model,
    verbose: int = 0
) -> GridSearchCV:
    '''Perfoms GridSearchCV on specified classifier or regressor pipeline.

    Args:
        dataset: PyrfumeDataset instance containing prediction target and molecule features.
        pipeline: Model instance specifies pipeline steps, prameter grid, and scoring.
        verbose: Indicates level of display output.

    Returns:
        GridSearchCV instance.
    '''
    assert dataset.task == pipeline.task, f'Conflicting tasks: {dataset.task} vs {pipeline.task}'

    features, targets = dataset.get_features_targets()
    scoring = pipeline.scoring
    cv = resolve_cv(dataset.task, dataset.n_splits)
    param_grid = reformat_param_grid(pipeline.param_grid)
    pipeline = make_pipeline(*pipeline.steps)

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit=False,
        cv=cv,
        verbose=verbose
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        gs.fit(features, targets)

    return gs


def evaluate_dummy_model(
        dataset: PyrfumeDataset,
        scoring: Union[str, List[str]] = None
) -> pd.DataFrame:
    '''Provides baseline results on dummy models for regression and classification.
    
    Args:
        dataset: PyrfumeDataset instance containing prediction target and molecule features.
        scoring: Scoring metric(s) to use. If None, will use ML task to get defaults.

    Returns:
        DataFrame with rows = strategies, cols = scoring.
    '''
    if dataset.task not in ['classification', 'regression']:
        raise ValueError('Must specify task as classification or regression.')
    if dataset.task == 'classification':
        strategies = ['most_frequent', 'prior', 'stratified', 'uniform']
        dummy_models = [
            DummyClassifier(strategy=s, random_state=RANDOM_STATE) for s in strategies
        ]
    elif dataset.task == 'regression':
        strategies = ['mean', 'median']
        dummy_models = [DummyRegressor(strategy=s) for s in strategies]

    features, targets = dataset.get_features_targets()
    scoring = resolve_scoring(scoring, dataset.task)
    cv = resolve_cv(dataset.task, dataset.n_splits)

    dummy_dict = {}
    for score in scoring:
        av_scores = [
            cross_val_score(model, features, targets, scoring=score, cv=cv).mean()
            for model in dummy_models
        ]
        dummy_dict[score] = av_scores

    return pd.DataFrame(dummy_dict, index=strategies).rename_axis('dummy_strategy')


def gridsearch_results_to_dataframe(
        gs_list: Union[GridSearchCV, List[GridSearchCV]]
) -> pd.DataFrame:
    '''Parse GridSearchCV results into DataFrame for easier downstream handling.'''
    if not isinstance(gs_list, List):
        gs_list = [gs_list]

    df_all = pd.DataFrame()

    for gs in gs_list:
        df = pd.DataFrame(gs.cv_results_)
        df['pipeline_steps'] = [
            list(gs.estimator.named_steps.values())] * len(df)
        df['pipeline_string'] = df['pipeline_steps'].apply(pipeline_steps_to_string)

        # Condense pipeline parameters into string
        df['param_string'] = df['params'].apply(
            lambda p: ';'.join([f"{k}={v}" for k, v in p.items()])
        )

        # Only keep mean and std of each scoring metric
        df = df[
            ['pipeline_string', 'pipeline_steps', 'param_string'] +
            [col for col in df.columns if any(
                map(col.__contains__, ['mean_test_', 'std_test_']))]
        ]
        df.columns = [col.replace('test_', '') for col in df.columns]

        df_all = pd.concat([df_all, df])

    return df_all.reset_index(drop=True)


def remove_prefix(text: str, prefix: str) -> str:
    '''For Python 3.8 compatibitity.'''
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_best_results(
        results: pd.DataFrame,
        metric: str = None,
        single_best: bool = False,
        include_pipeline_steps: bool = False
) -> pd.DataFrame:
    '''Filter out best scores and parameters for estimator pipelines.

    Args:
        resutls: GridSearchCV results from gridsearch_results_to_dataframe().
        metric: Scoring metric. If None, all scores and parameters are returned.
        single_best: Option to return single best result from all estimator pipelines.
            Only valid if a scoring metric is provided.
        include_pipeline_steps: Option to return list of steps in estimator pipeline.

    Returns:
        Dataframe of 'best' resutls from GridSearchCV.
    '''
    df = pd.melt(
        results.drop(columns='pipeline_steps'),
        id_vars=['pipeline_string', 'param_string'],
        value_vars=[col for col in results.columns if col.startswith('mean_')],
        var_name='metric',
        value_name='score'
    )
    df['metric'] = df['metric'].apply(lambda x: remove_prefix(str(x), 'mean_'))
    df = df.loc[df.groupby(['pipeline_string', 'metric'])['score'].idxmax()]
    df = df.pivot(index='pipeline_string', columns='metric', values=['score', 'param_string'])
    df = df.astype({col: float for col in df.columns if col[0] == 'score'})
    df.columns = [
        f'{tup[1]}_{tup[0]}'.replace('_score', '') for tup in df.columns.to_flat_index()
        ]
    df = df.join(results.set_index('pipeline_string')['pipeline_steps'].drop_duplicates())

    if metric is not None:
        df = df[[metric, f'{metric}_param_string', 'pipeline_steps']]
        if single_best:
            df = df.loc[df[metric].idxmax()].to_frame().T
    if not include_pipeline_steps:
        df.drop(columns='pipeline_steps', inplace=True)

    return df


def verify_batch_settings(
        archive: str,
        targets: List,
        feature_sets: List,
        prepare_dataset: Callable
    ):
    '''Verify that prediction target/feature set cobminations return valid datasets.
    
    Args:
        archive: Pyrfume-Data archive.
        targets: List of prediction targets.
        feature_sets: List of molecule feature types.
        prepare_dataset: Function that creates pipeline compatible dataset.
    '''
    for target, feature_set in itertools.product(targets, feature_sets):
        try:
            dataset = prepare_dataset(archive, target, feature_set)
            dataset.describe()
        except DatasetError:
            print(f'Could not create PyrfumeDataset for {target} and {feature_set}')


def batch_gridsearchcv(
        archive: str,
        targets: List[str],
        feature_sets: List[str],
        pipelines: List[Model],
        prepare_dataset: Callable,
        **kwargs
) -> pd.DataFrame:
    '''Performs GridSearchCV over target, feature, and estimator pipeline permutations.
    
    Args:
        archive: Pyrfume-Data archive.
        targets: List of prediction targets.
        feature_sets: List of molecule feature types.
        pipelines: List of Model instances.
        prepare_dataset: Function that creates pipeline compatible dataset.

    Returns:
        DataFrame of top scoring metircs and parameter sets for each combination of prediction
            target, feature set, and estimator pipeline.
    '''
    batch_results = {}
    for feature_set, target in itertools.product(feature_sets, targets):
        print(f'\nFeatures = {feature_set}, Prediction target = {target}\n{"-" * 50}')

        dataset = prepare_dataset(archive, target, feature_set)
        gs_list = []
        for pipeline in pipelines:
            print(f'\n{pipeline.pipeline_string}')
            gs_list.append(
                evaluate_model(dataset=dataset, pipeline=pipeline, **kwargs)
            )
        results = gridsearch_results_to_dataframe(gs_list)
        batch_results[(target, feature_set)] = get_best_results(
            results,
            include_pipeline_steps=True
        ).reset_index()

    df = pd.concat(batch_results, ignore_index=False).reset_index()
    df = df.rename(columns={'level_0': 'target', 'level_1': 'features'}).drop(columns='level_2')
    df = df.set_index(['target', 'features', 'pipeline_string']).sort_index()

    return df


def reconstruct_pipeline(pipeline_steps: List, param_string: str) -> Pipeline:
    '''Reconstruct Pipeline from list of steps and parameter string.
    
    Args:
        pipeline_steps: List of preprossing steps (optional) and estimator (must be last element
            in list).
        param_string: Parameters for estimator in condensed string form.

    Returns:
        sklearn Pipeline instance.
    '''
    params = {}
    for pair in param_string.split(';'):
        name, value = pair.split('=')
        if value.replace('.', '').isnumeric():
            if value.isdigit():
                value = int(value)
            else:
                value = float(value)
        params[name] = value
    pipeline = make_pipeline(*pipeline_steps)
    pipeline.set_params(**params)
    return pipeline


def reconstruct_model(
        dataset: PyrfumeDataset,
        pipeline_steps: List = None,
        param_string: str = None,
        results: pd.DataFrame = None,
        metric: str = None
) -> Pipeline:
    '''Reconstruct prediction-ready model.

    Must provide either pipeline steps and parameter string to reconstruct a specific model,
    or a results dataframe and scoring metric to reconstruct top-scoring model.
    
    Args:
        dataset: PyfumeDataset class instance.
        pipeline_steps: List of steps for pipeline. Last step must be an estimator.
        param_string: Parameters for estimator.
        resutls: Results for best GridSearchCV scoring metrics from either get_best_results()
            or batch_gridsearchcv().
        metric: Scoring metric which determines 'best' estimator.
    
    Returns:
        Prediction-ready Pipeline instance.
    '''
    # Reconstruct from provided steps and parameters
    if (pipeline_steps is not None) & (param_string is not None):
        pipeline = reconstruct_pipeline(
            pipeline_steps=pipeline_steps,
            param_string=param_string
        )
    # Reconstruct from top-scoring metric
    elif (results is not None) & (metric is not None):
        if metric not in results.columns:
            available_metrics = [
                col for col in results.columns
                if ('param_string' not in col) & (col != 'pipeline_steps')
            ]
            raise ValueError(f'Must specify metric from: {available_metrics}.')
        if 'pipeline_steps' not in results.columns:
            raise ValueError('Pipeline steps must be included in results DataFrame.')
        best = results.loc[results[metric].idxmax()].to_frame().T
        pipeline = reconstruct_pipeline(
            pipeline_steps=best['pipeline_steps'].values[0],
            param_string=best[f'{metric}_param_string'].values[0]
        )
    else:
        raise ValueError('Must povide either steps and param string or results and metric.')

    features, targets = dataset.get_features_targets()
    pipeline.fit(features, targets)

    return pipeline


def apply_model(
        dataset: PyrfumeDataset,
        model: Pipeline,
        label_encoder: LabelEncoder = None
    ) -> pd.DataFrame:
    '''Test prediction from estimator pipeline.
     
     Args:
        dataset: PyfumeDataset class instance.
        model: Pre-fitted Pipeline instance.

    Returns:
        DataFrame with original prediction targets and predictions.
    '''
    features, _ = dataset.get_features_targets()
    target_name = dataset.target_name

    df = dataset.df[[target_name]].copy()
    df['prediction'] = model.predict(features)

    # If label encoder is provided, inverse transform back to class names
    if label_encoder is None:
        label_encoder = dataset.label_encoder
    if label_encoder is not None:
        if  df[target_name].dtype == int:
            df[target_name] = label_encoder.inverse_transform(df[target_name])
        df['prediction'] = label_encoder.inverse_transform(df['prediction'])

    if dataset.task == 'classification':
        df['is_correct'] = df[target_name] == df['prediction']
    elif dataset.task == 'regression':
        df['%_error'] = df['prediction'].sub(df[target_name]).div(df[target_name])\
            .mul(100).round(2)

    out_text = '\n'.join([
        f'Archive: {dataset.archive}\n{"-" * 30}',
        f'Target: {dataset.target_name}, Features: {dataset.feature_set}',
        f'Pipeline: {", ".join(str(tup[1]) for tup in model.steps)}'
    ])
    if dataset.task == 'classification':
        out_text += f'\nCorrect predictions: {df["is_correct"].sum()}/{df.shape[0]}'
    print(out_text)

    return df


def plot_heatmap(
        results: pd.DataFrame,
        save_fig: bool = False,
        maxima: str = None,
        show_rect: bool = False
        ):
    '''Generate heatmap of top metric scores.

    Args:
        results: Results for best GridSearchCV scoring metrics from either get_best_results()
            or batch_gridsearchcv().
        save_fig: If True, heatmap will be saved as 'benchmarks.png'.
        maxima: Can be 'row', 'column', or None (default). Allows for easier inspection of
            leading metrics by model, or leading models by metric.
        show_rect: If True, rectangles are added to identify lead models.
    '''
    sns.set(font_scale=1.0)
    sns.set_style('dark')

    # Summary dataframe of just metric scores
    summary = results[
        [col for col in results.columns if ('param_string' not in col and col != 'pipeline_steps')]
    ]

    # Coordinates of maxima, looking row-wise, column-wise, or for single largest value
    if maxima == 'column':
        coords = list(zip(
            [summary.index.get_loc(row) for row in summary.idxmax(axis=0)], # Row maxima indices
            list(range(summary.shape[1])) # All column indices
        ))
    elif maxima == 'row':
        coords = list(zip(
            list(range(summary.shape[0])), # All row indices
            [summary.columns.get_loc(col) for col in summary.idxmax(axis=1)])) # Col maxima indices
    elif maxima is None: # Presume the single highest score is desired
        coords = [np.unravel_index(np.argmax(summary), summary.shape)]

    sns.heatmap(
        summary,
        annot=True,
        annot_kws={'size': 9},
        yticklabels=True,
        cbar_kws={'label': 'Score'}
    )
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.ylabel('')
    plt.tight_layout()
    yticklabels = [ytl.get_text().replace('-', ', ') for ytl in plt.gca().get_yticklabels()]
    plt.gca().set_yticklabels(yticklabels)

    # Draw rectangles on max values
    if show_rect:
        for x, y in coords:
            plt.gca().add_patch(
                patches.Rectangle((y, x), width=1, height=1,  lw=2, ec='blue', fc='None')
            )

    if save_fig:
        plt.savefig('benchmarks.png', format='png')

    plt.show()


def save_benchmarks(results: pd.DataFrame):
    '''Save best metric scores for each pipeline to CSV file.

    Args:
        results: Results for best GridSearchCV scoring metrics from either get_best_results()
            or batch_gridsearchcv().
    '''
    if 'pipeline_steps' in results:
        results.drop(columns='pipeline_steps', inplace=True)

    results.to_csv('benchmarks.csv')
    print('benchmarks.csv saved')


def plot_score_report(results: pd.DataFrame, scoring_metrics: Union[str, List]):
    '''Make a strip plot of models x scores.'''
    if not isinstance(scoring_metrics, list):
        scoring_metrics = [scoring_metrics]

    scores = results.reset_index()[['pipeline_string'] + scoring_metrics].rename(
        columns={'pipeline_string': 'estimator'}
    )
    scores['estimator'] = scores['estimator'].str.split(';').str[-1]
    scores = scores.sort_values(
        by=scoring_metrics[0],
        ascending=False
    ).reset_index(drop=True).reset_index()

    sns.set_theme(style='whitegrid')

    pair_grid = sns.PairGrid(
        scores,
        x_vars=scoring_metrics, y_vars='index',
        hue = 'estimator',
        height=6, aspect=0.3
        )
    pair_grid.map(sns.stripplot, size=4, orient='h', jitter=0.01)

    for ax, title, xmin, xmax in zip(
        pair_grid.axes.flat,
        scoring_metrics,
        scores[scoring_metrics].min().values,
        scores[scoring_metrics].max().values
    ):
        ax.set_title(title, fontdict={'fontsize': 10})
        ax.set_yticklabels('')
        ax.tick_params(labelsize=8)
        ax.set_xlabel('score', fontdict={'fontsize': 10})
        ax.set_ylabel('')
        ax.set(xlim=(0.85 * xmin, 1.15 * xmax))
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    sns.despine(left=True, bottom=True)


def show_confusion_matrix(
        dataset: PyrfumeDataset,
        model: Pipeline,
        normalize: str = None
        ):
    '''Show confusion matrix given a model and Pyrfume dataset.'''
    X, y = dataset.get_features_targets()

    model.fit(X, y)
    y_predict = model.predict(X)
    c_mat = confusion_matrix(y, y_predict, normalize=normalize)

    plt.subplots(figsize=(6, 4))
    plt.imshow(c_mat, cmap='Blues')
    plt.colorbar(shrink=0.6)
    plt.grid(False)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')


def show_roc_curve(dataset: PyrfumeDataset, model: Pipeline):
    '''Show ROC curve(s) given model and Pyrfume dataset'''    
    X, y = dataset.get_features_targets()
    model.fit(X, y)

    # Predict the class probabilities
    y_pred_prob = model.predict_proba(X)

    # Binarize the labels
    y_test_bin = label_binarize(y, classes=np.unique(y))
    roc_curves = []
    auc_val = []
    fprs_all = []
    common_thresholds = np.linspace(0, 1, 100)

    # Compute the macro-averaged ROC curve and auc
    for i in range(y_test_bin.shape[1]): # iterate over the unique classes
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        interpolated_fpr = np.interp(common_thresholds, tpr, fpr)
        auc_val.append(auc(fpr, tpr))
        roc_curves.append((interpolated_fpr, common_thresholds))
        fprs_all.append(interpolated_fpr)

    # Convert ROC curves to longform df for seaborn
    data = pd.DataFrame(data=fprs_all).T
    data['tpr'] = common_thresholds
    data = pd.melt(data, id_vars='tpr', var_name='class', value_name='fpr')

    sns.set_style('white')
    sns.relplot(data=data, x='fpr', y='tpr', hue='class', kind="line", height=4, aspect=1.0)


def show_regressor_performance(
    dataset: PyrfumeDataset,
    model: Pipeline,
    plot_type: str = 'both'
    ):
    '''Plot residuals and actual v. predicted for a regression model'''
    X, y = dataset.get_features_targets()
    model.fit(X, y)

    # Fit the given estimator on the given params
    y_predict = model.predict(X)

    # Simple DataFrames of residuals to facilitate plotting.
    act_pred = pd.DataFrame({'actual': y, 'predicted': y_predict})
    residuals = pd.DataFrame({'residuals': y_predict - y})

    if plot_type == 'both':
        _, ax = plt.subplots(1, 2)
    else:
        _, ax = plt.subplots()
        ax = [ax, ax]

    sns.set_style('white')
    color = sns.color_palette('tab10')[4]

    if plot_type in ['residuals', 'both']:
        sns.set_color_codes('muted')
        sns.histplot(data=residuals, x='residuals', kde=True, ax=ax[0], color=color)

    if plot_type in ['actual_predicted', 'both']:
        sns.scatterplot(x='actual', y='predicted', data=act_pred, ax=ax[1], color=color)
        ax[1].plot(y, y, ls='--', dashes=(5, 5), color='k')
    plt.tight_layout()
