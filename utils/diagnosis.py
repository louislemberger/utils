import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.utils.validation import check_is_fitted


def baseline_comparison(df_perf, df_baseline, greater_is_better=False):
    """compares model performance with baseline


    Parameters
    ----------
    greater_is_better : bool, default=True
        whether the metric used is a score function (default), higher is good,
        or a loss function (False), in which case the sign of the difference is flipped.

    Returns
    -------
    pd.DataFrame
        columns:
            - difference: between model and baseline predictions
            - uplift: difference *100 / model

    Note:
    -----
    to be used in conjunction with model_performance()

    """

    df_result = pd.concat([df_perf.to_frame('model'),
                           df_baseline.to_frame('baseline')
                           ], 1)

    df_result['difference'] = df_result['model'] - df_result['baseline']

    df_result['difference'] *= 1 if greater_is_better else -1

    df_result['uplift_percent'] = df_result['difference'] * 100 / df_result['baseline']
    return df_result



def plot_metrics_keras(history):
    all_metrics = list(history.history.keys())
    # pair val_xxx and xxx together
    train_labels = [el for el in all_metrics if 'val_' not in el]
    val_labels = [el for el in all_metrics if 'val_' in el]

    Tot = len(all_metrics) // 2
    Cols = 2

    Rows = Tot // Cols
    Rows += Tot % Cols
    Position = range(1,Tot + 1)

    fig = plt.figure(figsize=(15,12))
    axes = []
    for k in range(Tot):
        ax = fig.add_subplot(Rows,Cols,Position[k])
        ax.plot(history.history[train_labels[k]], label='train')
        ax.plot(history.history[val_labels[k]], label='validation')
        ax.set_xlabel('epoch', fontsize=15)
        ax.set_ylabel(train_labels[k], fontsize=15)
        ax.legend()
        axes.append(ax)

    return axes

def class_proba_dist(df_summary):
    subset_train = df_summary[df_summary['sample'] == 'train']
    subset_test = df_summary[df_summary['sample'] == 'test']

    unpack_cols = ['actual', 'pred']
    y_train, y_pred_train = subset_train[unpack_cols].values.T
    y_test, y_pred_test = subset_test[unpack_cols].values.T
    index_train_pos, index_train_neg  = (y_train == 1), (y_train == 0)
    index_test_pos, index_test_neg = (y_test == 1), (y_test == 0)

    f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(15,10), sharey='row')
    pd.Series(y_pred_train[index_train_neg]).hist(alpha=0.5, label='0', ax=ax1)
    pd.Series(y_pred_train[index_train_pos]).hist(alpha=0.5, label='1', ax=ax1)
    ax1.set_ylabel('count')
    ax1.set_title('train')
    ax1.legend()
    pd.Series(y_pred_test[index_test_neg]).hist(alpha=0.5, label='0', ax=ax2)
    pd.Series(y_pred_test[index_test_pos]).hist(alpha=0.5, label='1', ax=ax2)
    ax2.set_title('test')
    ax1.set_ylabel('count')
    ax2.legend();

    pd.Series(y_pred_train[index_train_neg]).hist(alpha=0.5, label='0', ax=ax3, density=True)
    pd.Series(y_pred_train[index_train_pos]).hist(alpha=0.5, label='1', ax=ax3, density=True)
    ax3.set_xlabel('classifier output probability')
    ax3.set_ylabel('norm. count (arb. unit)')
    ax3.legend()
    pd.Series(y_pred_test[index_test_neg]).hist(alpha=0.5, label='0', ax=ax4, density=True)
    pd.Series(y_pred_test[index_test_pos]).hist(alpha=0.5, label='1', ax=ax4, density=True)
    ax4.set_xlabel('classifier output probability')
    ax4.legend();
    plt.show()

def plot_residuals(df_summary, kde=False):
    subset_train = df_summary[df_summary['sample'] == 'train']
    subset_test = df_summary[df_summary['sample'] == 'test']

    unpack_cols = ['actual', 'pred', 'percent_delta']
    y_train, y_pred_train, pdelta_train = subset_train[unpack_cols].values.T
    y_test, y_pred_test, pdelta_test = subset_test[unpack_cols].values.T

    f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_pred_train, y_train - y_pred_train, alpha=0.5, label='train');
    ax1.scatter(y_pred_test, y_test - y_pred_test, alpha=0.5, label='test');
    ax2.scatter(y_pred_train, pdelta_train, alpha=0.5, label='train');
    ax2.scatter(y_pred_test, pdelta_test, alpha=0.5, label='test');
    ax1.set_title('residuals (scatter)')
    ax2.set_title('residuals (scatter)')
    ax1.set_ylabel('actual - predicted difference')
    ax2.set_ylabel('percentage difference')
    ax1.legend()
    ax2.legend()
    import seaborn as sns
    f2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    if kde:
        sns.distplot(y_train - y_pred_train, label='train', ax=ax3);
        sns.distplot(y_test - y_pred_test, label='test', ax=ax3);
        sns.distplot(pdelta_train, label='train', ax=ax4);
        sns.distplot(pdelta_test, label='test', ax=ax4);
    else:
        ax3.hist(y_train - y_pred_train, alpha=0.5, bins=40, label='train');
        ax3.hist(y_test - y_pred_test, alpha=0.5, bins=40, label='test');
        ax4.hist(pdelta_train, alpha=0.5, bins=40, label='train');
        ax4.hist(pdelta_test, alpha=0.5, bins=40, label='test');

    ax3.set_title('residuals (distribution)')
    ax4.set_title('residuals (distribution)')
    ax3.set_xlabel('actual - predicted difference')
    ax4.set_xlabel('percentage difference')
    ax3.legend()
    ax4.legend()



def df_regression_summary(y_train, y_test, y_pred_train, y_pred_test):
    """collates information about model predictions

    Returns
    -------
    df_summary: pd.DataFrame

    """
    df_summary = pd.DataFrame({'actual':np.concatenate([y_train, y_test]),
                               'pred': np.concatenate([y_pred_train, y_pred_test]),
                               'sample':['train'] * y_train.shape[0] + ['test'] * y_test.shape[0]},
                              index=np.concatenate([y_train.index.values, y_test.index.values])
                              )
    df_summary['residuals'] = df_summary['actual'] - df_summary['pred']
    df_summary['percent_delta'] = df_summary['residuals'] / df_summary['actual'] * 100
    return df_summary


def model_performance(model, metric_func, X_train, X_test, y_train, y_test, pred_out=False):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    perf_train = metric_func(y_train, y_pred_train)
    perf_test = metric_func(y_test, y_pred_test)
    df_perf = pd.Series([perf_train, perf_test], index=['train', 'test'])

    if pred_out:
        return y_pred_train, y_pred_test, df_perf
    else:
        return df_perf

def gini_coeff(y_true, y_pred, sample_weight=None):
    "Calculate gini coefficient from auroc"

    if isinstance(sample_weight, str) and (sample_weight=='balanced'):
            sw = y_true.shape[0] / (2 * np.bincount(y_true.astype(np.int64)))
            sample_weight = y_true.map({0:sw[0], 1:sw[1]}).values

    return 2 * roc_auc_score(y_true, y_pred, sample_weight=sample_weight) - 1

def pr_curve(y_true, y_pred, show_neg=False):
    "precision recall curve with iso-f1 lines"

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    if show_neg:
        precision_neg, recall_neg, _ = precision_recall_curve(1-y_true, 1-y_pred)
        plt.plot(recall, precision, label='positive class')
        plt.plot(recall_neg, precision_neg, label='negative class')
        plt.legend()
    else:
        plt.plot(recall, precision, color='b')

    f_scores = np.linspace(0.2, 0.8, num=4)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate(f'f1={f_score:0.1f}', xy=(0.9, y[45] + 0.02))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontweight='bold', fontsize=12)
    plt.ylabel('Precision', fontweight='bold', fontsize=12)
    plt.show()

def calc_lift(y_true, y_pred, N=10):

    df_lift = pd.DataFrame({'actual': y_true, 'pred': y_pred})
    df_lift['dummy_no'] = 1
    df_lift['quantile'] = pd.qcut(df_lift.pred, N, duplicates='drop', labels=False)
    df_lift['reverse_quantile'] = N-1 - df_lift['quantile']
    df_quantile = df_lift.groupby('reverse_quantile')['dummy_no'].sum().cumsum()
    df_quantile = df_quantile.to_frame('#contacted')
    df_quantile['#positive'] = df_lift.groupby('reverse_quantile').actual.sum().cumsum()
    df_quantile['%contacted'] = df_quantile['#contacted'] / df_quantile['#contacted'].max() * 100
    df_quantile['%positive'] = df_quantile['#positive'] / df_quantile['#positive'].max() * 100
    df_quantile['lift'] = df_quantile['%positive'] / df_quantile['%contacted']
    return df_quantile

def lift_curve(y_true, y_pred, N=10, show_neg=False, figsize=(7, 7)):
    "lift curve"


    df_quantile = calc_lift(y_true, y_pred, N)
    plt.figure(figsize=figsize)
    if show_neg:
        df_quantile_neg = calc_lift(1-y_true, 1-y_pred, N)
        plt.plot(df_quantile['%contacted'], df_quantile['lift'], label='positive class')
        plt.plot(df_quantile_neg['%contacted'], df_quantile_neg['lift'], label='negative class')

    else:
        plt.plot(df_quantile['%contacted'], df_quantile['lift'], label='model')

    plt.axhline(1, ls='--', c='r', label='no model')
    plt.ylabel('lift', fontsize=15, fontweight='bold')
    plt.xlabel('%population identified', fontsize=15, fontweight='bold')
    plt.legend()
    plt.show()

def gain_chart(y_true, y_pred, show_neg=False, figsize=(7,7)):

    df_lift = calc_gain(y_true, y_pred)
    plt.figure(figsize=figsize)
    if show_neg:
        df_lift_neg = calc_gain(1-y_true, 1-y_pred)
        plt.plot(df_lift['cumsum_all'], df_lift['cumsum_actual'], label='positive class')
        plt.plot(df_lift_neg['cumsum_all'], df_lift_neg['cumsum_actual'], label='negative class')
    else:
        plt.plot(df_lift['cumsum_all'], df_lift.cumsum_actual, label='model')
    plt.plot([df_lift['cumsum_all'].min(), df_lift['cumsum_all'].max()],
             [df_lift['cumsum_actual'].min(), df_lift['cumsum_actual'].max()], label='random', ls='--')
    plt.legend()
    plt.xlabel('% population', fontsize=15, fontweight='bold')
    plt.ylabel('% class' if show_neg else '%positive class', fontsize=15, fontweight='bold')
    plt.title('Cumulative gains chart', fontsize=15, fontweight='bold')
    plt.show()

def calc_gain(y_true, y_pred):

    df_lift = pd.DataFrame({'actual': y_true, 'pred': y_pred})
    df_lift = df_lift.sort_values('pred', ascending=False)
    df_lift['cumsum_actual'] = df_lift.actual.cumsum() / df_lift.actual.sum() * 100
    df_lift['dummy_no'] = 1
    df_lift['cumsum_all'] = df_lift['dummy_no'].cumsum() / df_lift.dummy_no.sum() * 100

    return df_lift

def roc(y_true, y_pred, show_neg=False):
    "unweighted auroc"

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    gini = gini_coeff(y_true, y_pred)

    if show_neg:
        fpr_neg, tpr_neg, _ = roc_curve(1-y_true, 1-y_pred)
        plt.plot(fpr, tpr, label='positive class')
        plt.plot(fpr_neg, tpr_neg, label='negative class')
    else:
        plt.plot(fpr, tpr, label='model')

    print(f'auroc: {round(roc_auc_score(y_true, y_pred), 4)}')
    print(f'gini: {round(gini,4)}')
    plt.plot([0, 1], [0, 1], ls='--', label='random')
    plt.legend()
    plt.show()


