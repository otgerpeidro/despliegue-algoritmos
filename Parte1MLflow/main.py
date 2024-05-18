import argparse
import subprocess

def main(nombre_experimento, n_estimators):
    # Ejecuta el script de exploración de datos
    print("Ejecutando exploracion_datos.py...")
    subprocess.run(['python', 'exploracion_datos.py'])
    
    # Ejecuta el script de preprocesamiento de texto
    print("Ejecutando preprocesamiento_texto.py...")
    subprocess.run(['python', 'preprocesamiento_texto.py'])
    
    # Ejecuta el script de entrenamiento del modelo
    print("Ejecutando entrenamiento_modelo.py...")
    subprocess.run(['python', 'entrenamiento_modelo.py', '--nombre_experimento', nombre_experimento, '--n_estimators', str(n_estimators)])
    
    # Inicia MLflow y ngrok
    print("Ejecutando iniciar_mlflow_ngrok.py...")
    subprocess.run(['python', 'iniciar_mlflow_ngrok.py'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento de un modelo de clasificación de documentos con Scikit-learn y MLflow')
    parser.add_argument('--nombre_experimento', type=str, required=True, help='Nombre del experimento en MLflow')
    parser.add_argument('--n_estimators', type=int, required=True, help='Número de estimadores para el RandomForestClassifier')
    args = parser.parse_args()
    
    main(args.nombre_experimento, args.n_estimators)
