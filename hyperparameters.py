from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils import rmse

import numpy as np

hyperparameters = {
  'ridge' : {
    'alpha' : np.linspace(0, 1, 100)   
  },
  'lasso' : {
    'alpha' : np.linspace(0, 1, 100)
  },
  'elasticnet' : {
    'alpha' : np.linspace(0, 1, 100)
  },
  'knn' : {
    'n_neighbors' : range(1, 11),
    'weights' : ['uniform', 'distance']
  },
  'randomforest' : {
    'n_estimators' : [5000, 10000, 150000, 30000],
    'criterion' : ['squared_error', 'absolute_error'],
    'max_depth' : [3, 4, 5, 6]
  },
  'gradientboosting' : {
    'n_estimators' : [5000, 10000, 150000, 30000],
    'loss' : ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'max_depth' : [3, 4, 5, 6]
  },
  'svr' : {
    'kernel' : ['linear', 'poly', 'rbf'],
    'degree' : [i for i in range(3, 11)],
    'C' : [.25, .5, .75, 1.]
  },
  'mlp' : {
    'hidden_layer' : [(16, 16), (16, 32), (32, 32), (32, 64), (64, 64), (16, 64), (64, 128)],
    'alpha' : [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
    'lr' : [0.01, 0.005, 0.001, 0.0005, 0.0001]
  }
}

def get_best_hparams(model_name, x_tr, x_ts, y_tr, y_ts):
  best_hparam = {}
  if model_name in ['ridge', 'lasso', 'elasticnet']:
    
    temp = 0
    for a in hyperparameters[model_name]['alpha']:
      if model_name == 'ridge':
        model = Ridge(alpha=a).fit(x_tr, y_tr)
      if model_name == 'lasso':
        model = Lasso(alpha=a, max_iter=3000).fit(x_tr, y_tr)
      if model_name == 'elasticnet':
        model = ElasticNet(alpha=a, max_iter=10000).fit(x_tr, y_tr)
        
      pred_tr = model.predict(x_tr)
      pred_ts = model.predict(x_ts)
      
      train_r2, test_r2 = r2_score(y_tr, pred_tr), r2_score(y_ts, pred_ts)
      train_rmse, test_rmse = rmse(y_tr, pred_tr), rmse(y_ts, pred_ts)
      
      if test_r2 > temp:
        temp = test_r2
        best_hparam = model.get_params()
  
  elif model_name == 'knn':
    temp = 0
    for a in hyperparameters[model_name]['n_neighbors']:
        for w in hyperparameters[model_name]['weights']:
            model = KNeighborsRegressor(n_neighbors=a, weights=w).fit(x_tr, y_tr)
            pred_tr = model.predict(x_tr)
            pred_ts = model.predict(x_ts)

            train_r2, test_r2 = r2_score(y_tr, pred_tr), r2_score(y_ts, pred_ts)
            train_rmse, test_rmse = rmse(y_tr, pred_tr), rmse(y_ts, pred_ts)

            if test_r2 > temp:
                temp = test_r2
                best_hparam = model.get_params()
                
  elif model_name == 'randomforest':
    temp = 0

    for n in hyperparameters[model_name]['n_estimators']:
        for c in hyperparameters[model_name]['criterion']:
            for d in hyperparameters[model_name]['max_depth']:
                model = RandomForestRegressor(n_estimators=n, criterion=c, max_depth=d,
                random_state=1, n_jobs=-1).fit(x_tr, y_tr)

                pred_tr = model.predict(x_tr)
                pred_ts = model.predict(x_ts)

                train_r2, test_r2 = r2_score(y_tr, pred_tr), r2_score(y_ts, pred_ts)
                train_rmse, test_rmse = rmse(y_tr, pred_tr), rmse(y_ts, pred_ts)

                if test_r2 > temp:
                    temp = test_r2
                    best_hparam = model.get_params()
                    
  elif model_name == 'gradientboosting':
    temp = 0

    for n in hyperparameters[model_name]['n_estimators']:
        for c in hyperparameters[model_name]['loss']:
            for d in hyperparameters[model_name]['max_depth']:
                model = GradientBoostingRegressor(n_estimators=n, loss=c, max_depth=d,
                random_state=1, validation_fraction=0.1, n_iter_no_change=50).fit(x_tr, y_tr)

                pred_tr = model.predict(x_tr)
                pred_ts = model.predict(x_ts)

                train_r2, test_r2 = r2_score(y_tr, pred_tr), r2_score(y_ts, pred_ts)
                train_rmse, test_rmse = rmse(y_tr, pred_tr), rmse(y_ts, pred_ts)

                if test_r2 > temp:
                    temp = test_r2
                    best_hparam = model.get_params()
    
  elif model_name == 'svr':
    temp = 0

    for k in hyperparameters[model_name]['kernel']:
        for c in hyperparameters[model_name]['C']:
            if k == 'poly':
                for d in hyperparameters[model_name]['degree']:
                    model = SVR(kernel=k, C=c, degree=d).fit(x_tr, y_tr)

                    pred_tr = model.predict(x_tr)
                    pred_ts = model.predict(x_ts)

                    train_r2, test_r2 = r2_score(y_tr, pred_tr), r2_score(y_ts, pred_ts)
                    train_rmse, test_rmse = rmse(y_tr, pred_tr), rmse(y_ts, pred_ts)

                    if test_r2 > temp:
                        temp = test_r2
                        best_hparam = model.get_params()                
            else:
                model = SVR(kernel=k, C=c).fit(x_tr, y_tr)
                pred_tr = model.predict(x_tr)
                pred_ts = model.predict(x_ts)

                train_r2, test_r2 = r2_score(y_tr, pred_tr), r2_score(y_ts, pred_ts)
                train_rmse, test_rmse = rmse(y_tr, pred_tr), rmse(y_ts, pred_ts)

                if test_r2 > temp:
                    temp = test_r2
                    best_hparam = model.get_params()

  elif model_name == 'mlp':
    temp = 0

    X_tr = np.array(x_tr.values)
    X_ts = np.array(x_ts.values)
    
    for h in hyperparameters[model_name]['hidden_layer']:
      for a in hyperparameters[model_name]['alpha']:
          for l in hyperparameters[model_name]['lr']:
              model = MLPRegressor(
                  hidden_layer_sizes=h,
                  activation='relu',
                  solver='adam',
                  alpha=a,
                  learning_rate_init=l,
                  max_iter=3000,
                  learning_rate='adaptive',
                  shuffle=True,
                  random_state=0,
                  early_stopping=True,
                  n_iter_no_change = 10
              ).fit(X_tr, y_tr)

              pred_tr = model.predict(X_tr)
              pred_ts = model.predict(X_ts)

              train_r2, test_r2 = r2_score(y_tr, pred_tr), r2_score(y_ts, pred_ts)
              train_rmse, test_rmse = rmse(y_tr, pred_tr), rmse(y_ts, pred_ts)

              if test_r2 > temp:
                  temp = test_r2
                  best_hparam = model.get_params()
                  
  return best_hparam
