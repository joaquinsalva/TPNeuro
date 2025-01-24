# Importo librerías y DataSets
from main import *

# GAUSS - variables

creator1 = PipelineCreator(problem_type="classification")
creator1.add("zscore")
creator1.add(“select_percentile”, percentile=[1,3,5,9,12,20,50,100])
creator1.add("gauss", kernel=[RBF(), Matern(), RationalQuadratic()])

creators = [creator1]

# Crear carpeta para guardar los datos

if not os.path.exists('gauss'):
        os.makedirs('gauss')  

# Plotear como varían los scores de los folds según random split

def scores_por_fold(all, all_names, creator, columns, nom):
    set_scores = []
    for j in range(len(all)):
        X = columns
        y = "group"
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        
        scores, model, inspector = run_cross_validation(
                X=X, 
                y=y, 
                data=all[j],
                model=creator,
                return_train_score=True,
                return_inspector=True, 
                cv=cv,
        )
        best_model = model.best_params_
        best_score = model.best_score_

        set_scores.append((all_names[j], scores, best_model, best_score))

    results = pd.DataFrame(set_scores, columns=['Dataset', 'Scores', 'best_model', 'best_score'])
    camino ='./gauss/' +  nom +'_gauss'+ '.csv'
    results.to_csv(camino, index=False) 


scores_por_fold(rois_all, rois_all_names, creator1, rois_columns,'rois')
scores_por_fold(volume_all, volume_all_names, creator1, volume_columns, 'volume')
scores_por_fold(thickness_all, thickness_all_names, creator1, thickness_columns, 'thickness')

