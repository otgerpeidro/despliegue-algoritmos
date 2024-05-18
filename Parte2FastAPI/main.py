from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Definición de los pipelines de Hugging Face
sentiment_pipeline = pipeline('sentiment-analysis')
qa_pipeline = pipeline('question-answering', model='BSC-LT/roberta-base-bne-sqac')

@app.get('/saludo')
def saludar():
    return {"message": "Hola, soy una API de FastAPI"}

@app.get('/sumar')
def sumar(a: int, b: int):
    return {"resultado": a + b}

@app.get('/concatenar')
def concatenar(a: str, b: str):
    return {"resultado": a + b}

@app.get('/sentimiento')
def sentimiento(texto: str):
    resultado = sentiment_pipeline(texto)
    return {"sentimiento": resultado[0]['label'], "score": resultado[0]['score']}

@app.get('/preguntar')
def preguntar(question: str, context: str):
    resultado = qa_pipeline(question=question, context=context)
    return {"respuesta": resultado['answer']}

# Ejemplo de datos para la pregunta y el contexto
contexto = '''
Los primeros coding bootcamps comenzaron en 2011 en los Estados Unidos enfocándose en el desarrollo web, tendencia que todavía continúa.
En julio de 2017, había 95 coding bootcamps a tiempo completo en los Estados Unidos. La duración de los cursos suele oscilar entre 8 y 36 semanas, y la mayoría dura de 10 a 12 semanas.
En el año 2013, los bootcamps llegaron a España. Desde entonces, se han consolidado como una de las fórmulas con las que especializarse en competencias STEM.
'''

@app.get('/ejemplo_pregunta')
def ejemplo_pregunta():
    pregunta = "¿Cuántos bootcamps había en 2017?"
    return preguntar(question=pregunta, context=contexto)
