# Importamos las librerías necesarias
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargamos el conjunto de datos de 20 Newsgroups
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

# Convertimos el conjunto de datos en un DataFrame de pandas
df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})

# Inicializamos el vectorizador TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Aplicamos el vectorizador a los textos del DataFrame
X = vectorizer.fit_transform(df['text'])

# Convertimos la matriz TF-IDF a un DataFrame
X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Añadimos la columna 'target' al DataFrame TF-IDF
X_df['target'] = df['target']

# Guardamos el DataFrame TF-IDF en un archivo CSV
X_df.to_csv('preprocessed_data.csv', index=False)

print("El DataFrame TF-IDF se ha guardado correctamente en 'preprocessed_data.csv'")

# Mostramos una vista previa del DataFrame TF-IDF con la columna 'target'
print("\nVista previa del DataFrame TF-IDF con la columna 'target':")
print(X_df.head())