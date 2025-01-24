# Importo librerías y DataSets
from main import *

# modelos 
# KNN - variables 

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('feature_selection', SelectPercentile()),  # Aquí configura los parámetros de SelectPercentile según tus necesidades
    ('classification', KNeighborsClassifier())
])

parameters = {
    'classification__n_neighbors' : range(10, 80, 10),
    'classification__weights': ['uniform', 'distance'],
    'feature_selection__percentile': [1,3,5,9,12,20,50,100]
}

# Crear carpeta para guardar los datos

if not os.path.exists('knn'):
        os.makedirs('knn')  

# Definición de función: cambiando el random, todos los datos cómo varía la accuracy y best_params

def todos_los_datos(all, parameters, all_names, nom):
    set_scores = []
    for j in range(len(all)): 
        X = all[j].iloc[:, 1:]
        y = all[j]['group']

        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

        grid_search = GridSearchCV(estimator = pipeline,
                                    param_grid = parameters,
                                    scoring = 'accuracy',
                                    cv= cv, 
                                    return_train_score = True
        )
        grid_search.fit(X, y)

        results = pd.DataFrame(grid_search.cv_results_)
        # Obtener el mejor puntaje y los mejores parámetros
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

        set_scores.append((all_names[j], results, best_params, best_score))


    # Guardar el DataFrame results como un archivo CSV en una ruta específica
    results = pd.DataFrame(set_scores, columns=['Dataset', 'Results', 'best_params', 'best_score'])
    camino = './knn/' + nom + '_knn'+ '.csv'
    results.to_csv(camino, index=False)

todos_los_datos(rois_all, parameters, rois_all_names, 'rois')
todos_los_datos(volume_all, parameters, volume_all_names, 'volume')
todos_los_datos(thickness_all, parameters, thickness_all_names, 'thickness')
