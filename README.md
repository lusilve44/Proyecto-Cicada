# Proyecto-Cicada

El proyecto se basa en el tratamiento de casos de datos desbalanceados en la columna de interés a predecir. 

Primero se empieza por un paneo general del dataset y el problema. El dataset es sobre objetos cercanos a la tierra, neo´s por sus siglas en inglés (Near Earth Objects). 

Luego se exploran las distintas variables del dataset y sus distribuciones corresponientes, a la vez que otros aspectos de interés, como son la matriz de correlaciones, el vif (variance inflation factor) entre otros. Seguido de haber estudiado esto, se hacen las transformaciones necesarias para nuestros algoritmos.

Ahora si pasamos al análisis de las distintas maneras de enfrentarnos a este problema. En resumen se aplican los siguientes métodos: 

- LogisticRegression:     
    - Penalización
    - Subsampling
    - Oversampling
    - Smote-Tomek 

- Random forest: 
    - Class Weighting
    - Bootstrap Class Weighting
    - Random Undersampling
    - Ensemble 

- One class SVM 


Finalmente se realizan comparaciones y se hipertunean los parámetros de los mejores modelos encontrados