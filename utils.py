from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib import rcParams
import seaborn as sns
from scipy import stats
import numpy as np


def rmse(y, y_hat):
    mse = mean_squared_error(y, y_hat)
    return mse**0.5


def get_plot(pred_tr, pred_ts, name):
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.major.width'] = 1.5
    rcParams['ytick.major.width'] = 1.5
    rcParams['xtick.minor.width'] = 1.5
    rcParams['ytick.minor.width'] = 1.5
    rcParams['xtick.major.size'] = 6
    rcParams['ytick.major.size'] = 6
    rcParams['xtick.minor.size'] = 3
    rcParams['ytick.minor.size'] = 3
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    a = np.linspace(-3, 3, 100)
    ax.plot(a, a, color='k', label='$y = x$')
    ax.scatter(y_tr, pred_tr, edgecolor='k', facecolor='grey', alpha=0.8, label='Train data')
    ax.scatter(y_ts, pred_ts, edgecolor='k', facecolor='red', alpha=0.8, label='Test data')
    # plt.xlabel('True', fontsize=14)
    # plt.ylabel('Prediction', fontsize=14)
    # plt.legend(facecolor='white', fontsize=12, loc=2)
    
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_tick_params(direction='in', which='minor')
    ax.xaxis.set_tick_params(direction='in', which='major')
    ax.xaxis.set_major_formatter('{x:.1f}')
    # ax.xaxis.set_minor_formatter('{x:.1f}')

    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_tick_params(direction='in', which='minor')
    ax.yaxis.set_tick_params(direction='in', which='major')
    ax.yaxis.set_major_formatter('{x:.1f}')
    # ax.yaxis.set_minor_formatter('{x:.1f}')

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    


    ax.patch.set_facecolor('#F5FAF5')
    # ax.patch.set_alpha(0.85)
    plt.title(name)
    plt.plot()
    

def get_compare(y_ts, pred_ts):
    a = pd.DataFrame(data = y_ts, index=y_ts.index)
    b = pd.DataFrame(data=pred_ts, index=y_ts.index)
    c = pd.concat([a, b], axis=1)
    c.columns=['True', 'Predicted']
    c['Differ'] = np.abs(c['True'] - c['Predicted'])
    return c

def get_comprehensive(df, pred_ts, name):
    df[name] = pred_ts
    df[name+'_error'] = np.round(np.abs(df['Target'] - df[name]), 3)
    return df
  
  
  
def get_feature_importance(model, model_name):
  # Top 10 important features
  
  feat_imp =  model.feature_importances_
  sorted_idx = np.argsort(rf_feat_imp)
  sorted_idx = sorted_idx[-10:]
  pos = np.arange(sorted_idx.shape[0]) + 0.5

  fig = plt.figure(figsize=(6, 6))
  rcParams['axes.linewidth'] = 1.5
  rcParams['xtick.major.width'] = 1.5
  rcParams['xtick.minor.width'] = 1.5
  rcParams['xtick.major.size'] = 6
  rcParams['xtick.minor.size'] = 3
  ax = fig.add_subplot()
  plt.barh(pos, rf_feat_imp[sorted_idx], align='center', color='#bad4f5')
  plt.yticks(pos, x_tr.keys()[sorted_idx])

  ax.yaxis.set_ticks_position('none')

  ax.xaxis.set_minor_locator(MultipleLocator(0.05))
  ax.xaxis.set_tick_params(direction='in', which='minor')
  ax.xaxis.set_tick_params(direction='in', which='major')
  ax.xaxis.set_major_formatter('{x:.2f}')
  for tick in ax.xaxis.get_major_ticks():
      tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
      tick.label1.set_fontweight('bold')
      
  plt.title(f'{model_name} Feature Importance')
  plt.plot()
  
  
def corrfunc(x, y, **kwargs):
    r, p = stats.personr(x, y)
    ax = plt.gca()
    ax.annotate('{r}', xy=(0.05, 0.9), xycoords=ax.transAxes)

def annotate_colname(x, **kwargs):
    ax = plt.gca()
    ax.annotate(x.name, xy = (0.05, 0.9), xycoords=ax.transAxes, fontweight='bold')

def corrdot(x, y, **kwargs):
    corr_r = x.corr(y, 'pearson')
    corr_text = f'{corr_r:2.2f}'.replace('0.', '.')
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([0.5], [0.5], marker_size, [corr_r], alpha=0.8, cmap='summer', vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40
    ax.annotate(corr_text, [0.5, 0.5], xycoords='axes fraction', ha='center', va='center', fontsize=font_size)


def cor_matrix(df):
    g = sns.PairGrid(df[columns], palette=['red'])
    g.map_upper(sns.regplot, lowess=True, ci=False, line_kws={'color':'red', 'lw':1}, scatter_kws={'color':'black', 's':10})
    g.map_diag(sns.distplot, kde=True, kde_kws={'color':'red', 'cut':0.7, 'lw':1}, hist_kws={'histtype':'bar', 'lw':2, 'facecolor':'#6ee09a'})
    g.map_diag(annotate_colname)
    g.map_lower(corrdot)

    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')

    return g
