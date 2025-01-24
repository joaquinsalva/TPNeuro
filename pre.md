# Pre-analysis
--------------

## Objetivos

El objetivo del presente trabajo es distinguir pacientes con bipolaridad y esquizofrenia de pacientes sanos. Para esto, decidimos replicar y realizar ciertas modificaciones al estudio presentado por el paper “Evaluation of Machine Learning Algorithms and Structural Features for Optimal MRI- Based Diagnostic Prediction in Psychosis” disponible en Zenodo. El estudio original usa distintos modelos de Machine Learning para identificar estos grupos usando imágenes por resonancia magnética (MRI). 

A partir de este paper publicado, se buscará mejorar la eficiencia de los modelos implementados al  prestarle atención a la elección de los hiperparámetros y métodos de validación de la capacidad predictiva de los modelos. Se buscará también analizar si los modelos son generalizables a nuevos datos. 


## Materiales y Métodos

 ### Paper de referencia

El estudio incluyó un total de 383 individuos, de los cuales 128 individuos poseían un diagnóstico de esquizofrenia, 128 de bipolaridad y 127 eran individuos de control (p.3, sección sample). Por cada individuo se tiene una imagen estructural del cerebro extraída mediante resonancia magnética y T1-weighted sequence. Una vez adquiridas las imágenes, se les realizó un preprocesamiento que incluía la extracción del tejido no cerebral y las imágenes estructurales originales fueron llevadas al espacio MINI152 2mm. (p.3, sección sMRI data features).

El programa FreeSurfer fue usado para parcelar la estructura del cerebro en regiones de interés, corticales y subcorticales. También se usó el volumen y grosor cortical promedio separados por hemisferio y VBM de la materia blanca y gris como principales features del estudio. Estos features se usaron para entrenar los modelos. Se evaluó la capacidad predictiva habiéndo reducido la dimensionalidad o no y juntando o no los features nombrados. Como métodos para reducir la dimensionalidad se usó la técnica de PCA y el calculó del univariate t-value para cada feature, quedándose con el 1% de datos cuyos t-scores fueron más altos (pp.3-5, sección sMRI data features).

Los modelos de aprendizaje automático utilizados en el estudio fueron ridge, lasso, elastic net y L0 norm regularized logistic regressions, support vector classifier, regularized discriminant analysis, random forests y Gaussian process classifier. (p.6, sección sMRI data learning algorithms).

La performance de los modelos,  a excepción de GPC y RF, se calcularon par a par en los tres grupos de sujetos usando nested cross validation. También se evaluaron modelos que predijeran los tres grupos de forma simultánea. Como métrica se usó la accuracy y la curva ROC. En el cross validation externo se usaron 10 folds pero para el interno no está especificado. Además, la elección de los features fue considerada como un parámetro más. (pp.6-7, sección sMRI General procedure and cross validation scheme y Multi-class classifiers).

 ### Replicación 

Para poder comparar nuestros datos con los obtenidos, de forma de disminuir las posibles diferencias, utilizaremos el mismo dataset y los datos procesados serán los mismos. Usaremos los features de libre acceso y computados por los autores del paper.

Los modelos utilizados con sus respectivos hiperparámetros que usaremos son los siguientes: 

 - Support Vector Machine: C, Gamma, kernel.
   
 - Random forest : cantidad de árboles en el bosque, altura, bootstrap, mínimo número de variables en una hoja, mínimo número de muestras en cada hoja antes de dividir, máxima cantidad de muestras consideradas antes de dividir ,(information gain, gain ratio, gini).
   
 - Ridge Regression:  alpha
   
 - KNN:  tipo de distancia usada.
   
 - Gaussian process classifier: kernel.

La cantidad de features utilizados para entrenar los modelos será considerado también como un parámetro más. La selección de los features usados dentro de cada dataset estará definida por el cálculo del univariate t-value. Sin embargo, evaluaremos las diferencias al variar los siguientes hiperparametros: porcentaje de features con el que nos quedamos al elegir aquellos que obtengan un mejor resultado y el hiperparámetros percentile del cálculo de los univariate t-values. 

Por último, entrenaremos un modelo “stacked” usando los modelos mencionados anteriormente con el objetivo de mejorar la performance. Evaluaremos qué sucede al entrenar con los distintos dataSets que tenemos a disposición. 

Todos los modelos serán probados entrenando con los mismos dataSets usados en el estudio. Evaluaremos nuestros modelos con 5 fold cross validation repetido 5 veces, a diferencia del estudio donde se hace nested cross validation.  La división de datos, el entrenamiento del modelo y la evaluación de la capacidad predictiva será comparada para distintos random seeds con el objetivo de evaluar la capacidad de generalización de los modelos.

## Conclusión. 

Finalmente, compararemos nuestros resultados con aquellos obtenidos en el paper, veremos si los resultados para los modelos elegidos coinciden con aquellos en el documento y evaluaremos las posibles razones, si es que los resultados difieren significativamente, Además, analizaremos si fue posible mejorar la capacidad predictiva de los modelos o si, a pesar de las modificaciones hechas, no fue posible. Por último, evaluaremos la capacidad de generalización más allá de la eficiencia en la capacidad predictiva de los modelos 

