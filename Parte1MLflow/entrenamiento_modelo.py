import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

def entrenamiento_modelo(nombre_experimento, n_estimators):
    # Cargamos el DataFrame preprocesado desde el archivo CSV
    df = pd.read_csv('preprocessed_data.csv')
    # Separamos las características y la variable objetivo
    X = df.drop(columns=['target'])
    y = df['target']
    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Configuramos el modelo de clasificación Random Forest
    clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, class_weight='balanced', random_state=42)
    # Entrenamos el modelo
    clf.fit(X_train, y_train)
    # Hacemos predicciones en el conjunto de prueba
    y_pred = clf.predict(X_test)
    # Evaluamos el rendimiento del modelo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    # Configuramos MLflow para registrar los experimentos
    mlflow.set_experiment(nombre_experimento)
    with mlflow.start_run(run_name='RandomForest_Classification'):
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('min_samples_leaf', 2)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.sklearn.log_model(clf, 'random_forest_model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento del modelo de clasificación de documentos')
    parser.add_argument('--nombre_experimento', type=str, required=True, help='Nombre del experimento en MLflow')
    parser.add_argument('--n_estimators', type=int, required=True, help='Número de estimadores para el RandomForestClassifier')
    args = parser.parse_args()
    
    entrenamiento_modelo(args.nombre_experimento, args.n_estimators)
