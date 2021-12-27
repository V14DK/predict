import uvicorn
import pickle
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from for_parsing import TextNormalizer, GensimVectorizer, GensimLsi

with open('static/Normalizer.pkl', 'rb') as file:
    normalizer = pickle.load(file)

with open('static/Vectorizer.pkl', 'rb') as file:
    vect = pickle.load(file)

with open('static/Lsi.pkl', 'rb') as file:
    lsi_model = pickle.load(file)

with open('static/Model_SVC.pkl', 'rb') as file:
    model = pickle.load(file)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def main():
    return FileResponse('static/file.html')


@app.post('/tonality/')
async def get_tonality(message: Request):
    text = await message.body()
    text = text.decode('utf-8')
    text = normalizer.transform(text)
    text = vect.transform(text)
    text = lsi_model.transform(text)
    return model.predict(text)[0]


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)