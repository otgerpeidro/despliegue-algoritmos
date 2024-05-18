# Importamos las librerías necesarias
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Cargamos el conjunto de datos de 20 Newsgroups
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

# Convertimos el conjunto de datos en un DataFrame de pandas
df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})

# Mostramos una vista previa de los primeros registros del DataFrame
print("Vista previa de los primeros registros del DataFrame:")
print(df.head())

# Mostramos la forma del DataFrame para conocer el número de filas y columnas
print(f"\nDimensiones del DataFrame: {df.shape}")

# Mostramos las estadísticas descriptivas del DataFrame
print("\nEstadísticas descriptivas del DataFrame:")
print(df.describe())

# Mostramos los nombres de las categorías presentes en el conjunto de datos
print("\nNombres de las categorías en el conjunto de datos:")
print(newsgroups.target_names)

# Mostramos un conteo de documentos por categoría
print("\nConteo de documentos por categoría:")
print(df['target'].value_counts())

# Realizamos una visualización básica del contenido de los documentos
print("\nEjemplo de contenido de un documento:")
print(df['text'].iloc[0])
