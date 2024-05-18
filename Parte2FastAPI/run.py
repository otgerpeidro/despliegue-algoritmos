import nest_asyncio
from pyngrok import ngrok
import uvicorn

# Configuración de ngrok
ngrok.set_auth_token("2gdXrCX0L9zRaeb1pPGAKqOC8xT_5g4MitQHBXE4g8934gYgg")
public_url = ngrok.connect(8000).public_url
print(f"Public URL: {public_url}")

# Necesario para ejecutar uvicorn en el notebook
nest_asyncio.apply()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)