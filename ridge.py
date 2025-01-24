# Entrenador Ridge 
from main import *
r = "Ridge"

#Creamos carpetas para almacenar resultados
if not os.path.exists(r):
        os.makedirs(r)  
    
if not os.path.exists(r + "/rois"):
    os.makedirs(r + "/rois")  
if not os.path.exists(r + "/volume2"):
    os.makedirs(r + "/volume2")  
if not os.path.exists(r + "/thickness2"):
    os.makedirs(r + "/thickness2")  



#Definimos el pipeline
Ridge = PipelineCreator(problem_type="classification", apply_to="*")
Ridge.add("zscore")
Ridge.add('select_percentile', percentile=[1,3,5,9,12,20,50,100])
Ridge.add("ridge",
          alpha =  [0.001, 0.01, 0.1, 1, 10, 100, 1000],
          solver = ['auto', 'svd','cholesky']
          )

# Entrenador de modelos variando percentiles. Devuelve accuracies y mejores parametros.
def train_ridge(all, all_names, creators, columns):
    for i in range(len(all)): #Por cada grupo de interes
                crossVal = RepeatedKFold(n_splits = 5,n_repeats = 5,random_state= 42)
                scores, model, inspector = run_cross_validation(
                    X=  columns, 
                    y='group', 
                    data=all[i],
                    model=creators,
                    return_train_score=True,
                    return_inspector=True, 
                    cv=crossVal,
                )
                results = scores[['test_score', 'train_score']]
                best_score_ = model.best_score_
                best_params_ = model.best_params_
                results['best_score'] = best_score_
                results['best_params_'] = str(best_params_)
                results.to_csv(r + "/" + all_names[3]+ "/" + f'{all_names[i]}.csv',index =False)
                
    
# Entrenamos ridge con Rois Dataset
train_ridge(rois_all, rois_all_names, Ridge, rois_columns)        
     
bip = pd.read_csv("Ridge/rois/bip_cnt_rois.csv")
# Entrenamos ridge con Thickness Dataset
train_ridge(thickness_all, thickness_all_names, Ridge, thickness_columns)        

# Entrenamos ridge con Volume Dataset
train_ridge(volume_all, volume_all_names, Ridge, volume_columns)        
