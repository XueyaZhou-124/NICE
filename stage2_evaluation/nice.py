# type: ignore
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
import functools
# SMOTE过采样
from imblearn.combine import SMOTETomek
import shap

import pandas as pd
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from joblib import Parallel, delayed
import joblib
import argparse
import json

import hashlib
import time
import warnings
warnings.filterwarnings("ignore")

# GLOBAL VARIABLES
PHE_COLS = ['Morphology', 'typeMorphology', 'typeMorphology2'] # 数据集中的PHE相关列
LABEL_COL = 'typeMorphology' # 用于的建模y列
POS_LABEL = 'high_quality' # 正例标签
SEED = 1995
# 1995 lgr 0.806
np.random.seed(SEED)
random.seed(SEED)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=12)
    
    # data args
    parser.add_argument('--label_col', type=str, default='typeMorphology')
    parser.add_argument('--thereshold', type = float)
    parser.add_argument('--valid_ratio', type = float, default=1)
    parser.add_argument('--filter_type', type=str)
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--k_features', type = int, default=10)
    parser.add_argument('--n_c', type = float, default=0.95)
    parser.add_argument('--need_preprocess', action='store_true')

    args = parser.parse_args()
    return args


def pheTonum(phe, pos_label = 'high_quality'):
    label = np.array([ int(i == pos_label) for i in phe])
    return label


# data preprocessing
def sampleSelect(data, reads_num = None, thereshold = 4.5):
    if reads_num is None:
        return data

    df = reads_num.loc[(reads_num['reads_num'] >= 1000)] # 先去掉reads数过低的样本
    selected_sample = df.loc[np.log10(df['reads_num']) >= thereshold].index
    retain_sample = list(set(data.index.tolist()).intersection(set(selected_sample)))
    data = data.loc[retain_sample]
    print(f'sample filter: {data.shape[0]} | {reads_num.shape[0]}')
    return data


# NA筛选
def nafilter(df, valid_ratio = 1, invalid_value = None):
    sample_num = df.shape[0]
    label = df['label']
    df = df.drop('label', axis = 1)
    if invalid_value is not None: # 去掉
        print(f'invalid value {invalid_value}')
        valid_res =  df.loc[:, (df == invalid_value).sum(axis = 0) <= sample_num * (1-valid_ratio)]
    else:
        valid_res =  df.loc[:, df.isna().sum(axis = 0) <= sample_num * (1-valid_ratio)]


    print(f'invalid filter: {valid_res.shape[1]} | {df.shape[1]}')
    valid_res = pd.concat([valid_res, label], axis = 1)

    cols = valid_res.columns
    indices = valid_res.index

    if valid_res.isna().any().any():
        imputer = KNNImputer(n_neighbors=1, )
        valid_res = pd.DataFrame(imputer.fit_transform(valid_res))
        valid_res.index = indices
        valid_res.columns = cols

    X = valid_res.iloc[:,:-1]
    X = X.loc[:,(X ==0 ).sum() <= X.shape[0] * 0.1]
    valid_res = pd.concat([X, label], axis = 1)
    # valid_res = valid_res.loc[:, (valid_res.iloc[:,:-1] ==0 ).sum() <= (valid_res.shape[0]) * 0.1]
    print(f'zero filter: {valid_res.shape[1]} | {df.shape[1]}')

    return valid_res


# 行归一化
def zscore_row(row_np):
    row_mean = row_np.mean()
    row_std = np.std(row_np)
    new_row = (row_np - row_mean)/row_std
    return new_row

# CLR转换
def clr_transform(df_features):
    """
    df_features 每行样本，每列feature，值为正比例，且合为1
    返回CLR转换后的DataFrame
    """
    # 加极小值避免log0
    df = df_features + 1e-9
    g = df.apply(lambda x: np.exp(np.mean(np.log(x))), axis=1)  # 几何平均
    clr_df = np.log(df.div(g, axis=0))
    return clr_df


def data_preprocess(data_path, save_path, reads_num = None, thereshold = 4.5, valid_ratio = 1,):
    data = pd.read_csv(data_path, index_col=0)
    if reads_num is not None:
        reads_num = pd.read_csv(reads_num, header = None)
        reads_num = reads_num.set_index(0)
        reads_num.columns = ['reads_num']
    # label
    data['label'] = pheTonum(data[LABEL_COL], pos_label=POS_LABEL)
    # 去掉PHE相关列
    data = data.drop(PHE_COLS, axis = 1)

    print('original high-label ratio:', data['label'].mean()) # 正例样本比例
    print('ori data shape:', data.shape)

    # 样本筛选
    print('----sample filtering----')
    thereshold = thereshold # reads 阈值
    # 特征筛选
    valid_ratio = valid_ratio # 有效值在样本里的比例

    # data preprocess
    df = data
    if 'flen' in data_path: # 对flen无效值为0
        invalid_value = 0
    else:
        invalid_value = None
    
    df = sampleSelect(df, reads_num, thereshold = thereshold)
    df = nafilter(df, valid_ratio=valid_ratio, invalid_value=invalid_value)

    # 去掉性XY染色体
    df = df.loc[:, ~np.array([ 'chrX' in i for i in df.columns])]
    df = df.loc[:, ~np.array([ 'chrY' in i for i in df.columns])]

    if 'total'  in data_path:
        # 行归一化
        data = df.iloc[:,:-1]
        data = data.apply(zscore_row, axis=1)
        df.iloc[:,:-1] = data
        print('zscore done')


    # 针对分布组成数据进行CLR转化
    if ('end_motif' in data_path) or ( 'flen' in data_path):
        df.iloc[:,:-1] = clr_transform(df.iloc[:,:-1])
        print('clr done')

    data_filtered = df
    print('data filtered shape:', data_filtered.shape)
    label = data_filtered['label']
    print('data filtered pos-label ratio:', label.mean()) # 正例样本比例
    # 对预处理后的数据进行保存
    # 按照一定index顺序保存
    data_filtered = data_filtered.sort_index()
    data_filtered.to_csv(os.path.join(save_path, 'data_preprocessed.csv'))

    return data_filtered


def model_init(trial, clf_name):
    # 超参数初始化
    random_state = SEED

    if clf_name == 'RF':
        n_estimators = trial.suggest_categorical("rf_n_estimators", [30, 50, 100])
        max_depth = trial.suggest_int("rf_max_depth", 2, 10)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 5)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )

    elif clf_name == 'SVC':
        kernel = trial.suggest_categorical("svc_kernel", ['linear', 'rbf'])
        C = trial.suggest_loguniform("svc_C", 0.1, 10.0)
        clf = SVC(kernel=kernel, probability=True, C=C, random_state=random_state, class_weight='balanced')

    elif clf_name == 'XGB':
        params = {
                    'max_depth': trial.suggest_int('max_depth', 2, 5),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.1),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                    'n_estimators': trial.suggest_int('n_estimators', 30, 100)
                }

        clf = XGBClassifier(**params, random_state = random_state)

    elif clf_name == 'LGR':
        # 逻辑回归
        C = trial.suggest_float("C", 0.01, 10.0)

        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "liblinear" 
        l1_ratio = None

        # 构造 clf 参数
        kwargs = {
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "random_state": random_state,
            "max_iter": 1000,
        }
        if l1_ratio is not None:
            kwargs["l1_ratio"] = l1_ratio

        clf = LogisticRegression(**kwargs)


    elif clf_name == 'MLP':
        # 多层感知机
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(20,), (50,), (20, 20), (10,20)])
        alpha = trial.suggest_loguniform("alpha", 0.0001, 0.1)
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=2000,
            early_stopping=True,
            random_state=random_state        
        )
    
    return clf


def cv_iteration(X, y, train_index, val_index, k_features = 50, n_c = 20, resample = False,
                  clf = RandomForestClassifier(), select_func = f_classif):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    if k_features > 0:
        n_features = X_train.shape[1]
        # 特征选择（基于单变量方差分析）
        if n_features >= k_features:
            selector = SelectKBest(score_func=select_func, k=k_features)
            X_train = selector.fit_transform(X_train, y_train)
            X_val = selector.transform(X_val)

    #（PCA）
    if n_c > 0:
        decomposer = PCA(n_components=n_c)
        X_train = decomposer.fit_transform(X_train)
        X_val = decomposer.transform(X_val)

    if resample:
        smote = SMOTETomek(random_state=SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # 训练模型
    clf.fit(X_train, y_train)

    # 验证集结果
    y_pred_i = clf.predict(X_val)
    y_score_i = clf.predict_proba(X_val)[:, -1]
    y_val_i = y_val

    return (y_val_i, y_pred_i, y_score_i)


# 模型评估
def evaluate( y_true, y_pred, y_score):
    """二分类"""
    auroc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    res = {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'auroc': auroc, 'auprc':auprc}
    return res


def objective(trial, X_train, y_train, clf_name = 'RF', k_features = 20, n_c = 0.95, resample=True, save_oof = False):

    clf = model_init(trial, clf_name) # get clf
    cv = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True) # 内层CV
    res = [cv_iteration(X_train, y_train, train_index, val_index, n_c = n_c, clf = clf, k_features = k_features, resample = resample)
                for train_index, val_index in cv.split(X_train,y_train)]
            
    # 整理结果
    y_true, y_pred, y_score = [], [], []
    for y_true_i, y_pred_i, y_score_i in res:
        y_true.append(y_true_i)
        y_pred.append(y_pred_i)
        y_score.append(y_score_i)
    # 转换为 NumPy 数组
    y_true = np.concatenate(y_true, axis = 0)
    y_pred = np.concatenate(y_pred, axis = 0)
    y_score = np.concatenate(y_score, axis = 0)

    # 模型评估
    eval_res = evaluate(y_true, y_pred, y_score)
    score = eval_res['auroc']
    if save_oof:
        res = (y_true, y_pred, y_score)
        return score, res

    return score


def objective_detailed(best_trial, X_train, y_train, X_test, y_test, clf_name,resample=True, 
                       k_features = 20, n_c = 0.95):
    select_func = f_classif
    features_in = X_train.columns
    # 1. 用最佳超参数重新训练模型
    clf = model_init(best_trial, clf_name)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    feature_selected = features_in
    if k_features > 0:
        n_features = X_train.shape[1]
        # 特征选择（基于单变量方差分析）
        if n_features >= k_features:
            selector = SelectKBest(score_func=select_func, k=k_features)
            X_train = selector.fit_transform(X_train, y_train, )
            X_test = selector.transform(X_test)
            feature_selected = selector.get_feature_names_out(features_in)

    if n_c > 0:
        #（PCA）
        decomposer = PCA(n_components=n_c)
        X_train = decomposer.fit_transform(X_train)
        X_test = decomposer.transform(X_test)
    
    if resample:
        # SMOTE过采样
        smote = SMOTETomek(random_state=SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)


    # 7. 模型训练
    clf.fit(X_train, y_train)

    # 8. 模型预测
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, -1]
    y_true = y_test

    # 9.特征重要性
    if n_c == 0:
        feature_names = feature_selected
        feature_importance = get_feature_importance(clf, feature_names=feature_names, X=X_test)
    else:
        feature_importance = []

    return y_true, y_pred, y_score, feature_importance


def get_config_hash(config):
    """基于config内容生成一个唯一的hash"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_feature_importance(model, feature_names, X):
    """
    通用特征重要性计算函数
    参数：
        model: 训练完成的分类器对象
        feature_names: 特征名称列表
    返回：
        dict: 特征名称与重要性值的映射
    """
    # 树模型处理逻辑
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            raise AttributeError("当前树模型未实现feature_importances_属性")
    
    # 逻辑回归处理逻辑
    elif isinstance(model, LogisticRegression):
        coef = np.abs(model.coef_)
        importance = np.mean(coef, axis=0)  # 多分类场景取平均
    
    # 其他模型使用SHAP
    else:
        try:
            explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X) # nsample,nfeatures
            importance = shap_values.T # 转置一下
            # importance = np.mean(np.abs(shap_values), axis=0) # 这里SHAP取绝对值
        except Exception as e:
            raise RuntimeError(f"SHAP计算失败: {str(e)}")

    return dict(zip(feature_names, importance))

def main():
    args = get_args()
    
    config = {'n_c': args.n_c, 'k_features': args.k_features, 'resample': args.resample, 'valid_ratio': args.valid_ratio, 'thereshold': args.thereshold}

    n_c = config['n_c']
    k_features = config['k_features']
    resample = config['resample']
    valid_ratio = config['valid_ratio']
    thereshold = config['thereshold']
    
    data_path = args.data_path
    basename = os.path.basename(data_path)[:-4]

    config_hash = get_config_hash(config)
    timestamp = time.strftime('%b%d-%H-%M')
    save_dir = os.path.join(args.save_path, basename, f'{config_hash}_{timestamp}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'config.json'), 'w+') as f:
        # 保存参数配置文件
        f.write(json.dumps(config))
        f.write('\n')

    # 数据预处理
    if args.need_preprocess:
        if thereshold is None:
            data_filtered = data_preprocess(
                data_path = os.path.join(data_path),
                save_path = save_dir,
                reads_num = None,
                valid_ratio = valid_ratio,
                thereshold = thereshold,
            )
        else:
            data_filtered = data_preprocess(
                data_path = os.path.join(data_path),
                save_path = save_dir,
                reads_num = os.path.join(os.path.dirname(data_path), 'reads_num.csv'),
                valid_ratio = valid_ratio,
                thereshold = thereshold,
            )
    else:
        data_filtered = pd.read_csv(args.data_path, index_col = 0)
        assert 'label' in data_filtered.columns
        print('data filtered shape:', data_filtered.shape)
        label = data_filtered['label']
        print('data filtered pos-label ratio:', label.mean()) # 正例样本比例
    X = data_filtered.drop(['label'], axis=1)
    y = data_filtered['label']

    clf_names = ['XGB', 'MLP', 'LGR', 'SVC', 'RF']

    eval_res_dict = {clf_name:[] for clf_name in clf_names}
    all_res_dict = {clf_name:[] for clf_name in clf_names}
    oof_res_dict = {clf_name:[] for clf_name in clf_names}
    feature_importance_dict = {clf_name:[] for clf_name in clf_names}

    # 外层五折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        # 当前折数据
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 训练每个模型
        for clf_name in clf_names:
            # 创建调参目标函数,只使用训练集搜参
            objective_with_data = functools.partial(objective, X_train=X_train, y_train=y_train, clf_name=clf_name,
                                                    k_features=k_features, resample=resample)
            sampler = TPESampler(seed=SEED)  # Make the sampler behave in a deterministic way.
            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(objective_with_data, n_trials=args.n_trials, n_jobs=args.n_jobs) # 搜参
            
            # 用最佳参数在训练集上进行CV，保存oof结果用作之后的stacking
            _, oof = objective(study.best_trial, X_train=X_train, y_train=y_train, clf_name=clf_name,
                                k_features=k_features, resample=resample, save_oof=True)
            # 用最佳参数在整个训练集上训练，在测试集上评估
            y_true, y_pred, y_score, feature_importance = objective_detailed(study.best_trial, X_train, y_train, X_test, y_test,  clf_name=clf_name,
                                                            k_features=k_features, n_c=n_c, resample=resample)
            # 转换为 numpy 数组
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_score = np.array(y_score)

            # 评估指标
            eval_res = evaluate(y_true, y_pred, y_score)

            # 打印和记录
            print(f'Fold {i+1}:')
            print(f"Eval result: {eval_res}")
            # 保存结果
            eval_res_dict[clf_name].append(eval_res)
            all_res_dict[clf_name].append((y_true, y_pred, y_score))
            oof_res_dict[clf_name].append(oof)
            feature_importance_dict[clf_name].append(feature_importance)

    joblib.dump(eval_res_dict, os.path.join(save_dir, 'eval_res.pkl'))
    joblib.dump(oof_res_dict, os.path.join(save_dir, 'oof_res.pkl'))
    joblib.dump(all_res_dict, os.path.join(save_dir, 'all_res.pkl'))
    joblib.dump(feature_importance_dict, os.path.join(save_dir, 'feature_importance.pkl'))

   # 求出每个分类器的在五折之间的平均AUROC
    score = {}
    for clf_name in clf_names:
        y_true = np.concatenate([all_res_dict[clf_name][fold][0]  for fold in range(5)])
        y_pred = np.concatenate([all_res_dict[clf_name][fold][1]  for fold in range(5)])
        y_score = np.concatenate([all_res_dict[clf_name][fold][2]  for fold in range(5)])

        score[clf_name] = (evaluate(y_true, y_pred, y_score)['auroc'])

    best_model = max(score, key=score.get)
    best_score = score[best_model]

    print('best model performance:')
    print(f'{best_model} auroc: {best_score}')
    
    
if __name__ == '__main__':
    main()

