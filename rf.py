#Entrenador Random Forest
from main import * 
rforest = "RandomForest"

if not os.path.exists(rforest):
        os.makedirs(rforest)  
 
#Creamos carpetas para almacenar resultados
if not os.path.exists(rforest + "/rois"):
    os.makedirs(rforest + "/rois")  
if not os.path.exists(rforest + "/volume2"):
    os.makedirs(rforest + "/volume2")  
if not os.path.exists(rforest + "/thickness2"):
    os.makedirs(rforest + "/thickness2")  


#Definimos el pipeline
rf = PipelineCreator(problem_type="classification", apply_to="*")
rf.add("zscore")
rf.add( 'select_percentile', percentile=[1,3,5,9,12,20,50,100])
rf.add(
    "rf",
    n_estimators= range(70,161,30),
    max_depth= range(2,13,3),
    min_samples_leaf= range(2, 33,8),
    min_samples_split= range(3,30,6), 
)


    

# Entrenador de modelos variando percentiles. Devuelve accuracies y mejores parametros.
def train_rf(all, all_names, creators, columns):
    for i in range(len(all)): #Por cada grupo de interes
        crossVal = RepeatedKFold(n_splits = 5,n_repeats = 5,random_state= 42)
        scores, model, inspector = run_cross_validation(
            X=columns, 
            y='group', 
            data=all[i],
            model=creators,
            return_train_score=True,
            return_inspector=True, 
            cv=crossVal,
            )
        print(model)
        results = scores[['test_score', 'train_score']]
        best_score_ = model.best_score_
        best_params_ = model.best_params_
        results['best_score'] = best_score_
        results['best_params_'] = str(best_params_)
        results.to_csv(rforest + "/" + all_names[3] + "/" + f'{all_names[i]}.csv',index =False  )
          
    
# Entrenamos Random Forest con Rois Dataset
train_rf(rois_all, rois_all_names, rf, rois_columns)        

# Entrenamos Random Forest con Thickness Dataset
train_rf(thickness_all, thickness_all_names, rf, thickness_columns)        

# Entrenamos Random Forest con Volume Dataset
train_rf(volume_all, volume_all_names, rf, volume_columns)        
