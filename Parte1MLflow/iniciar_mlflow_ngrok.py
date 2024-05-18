import mlflow
from pyngrok import ngrok
import subprocess

# Configura el experimento en MLflow
mlflow.set_experiment('Document Classification')

# Inicia el servidor de MLflow
mlflow_ui_process = subprocess.Popen(['mlflow', 'ui', '--port', '5000'])
print("MLflow server is running on http://localhost:5000")

# Inicia ngrok
ngrok.set_auth_token('2gdXrCX0L9zRaeb1pPGAKqOC8xT_5g4MitQHBXE4g8934gYgg')  # Token de autenticación de ngrok
ngrok_tunnel = ngrok.connect(addr='5000', proto='http', bind_tls=True)
print(f'El tracking UI de MLflow está disponible en: {ngrok_tunnel.public_url}')