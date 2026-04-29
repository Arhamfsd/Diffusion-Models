from fastapi import FastAPI

app = FastAPI()


@app.get("/generate")
def generate():
    return {"message": "This is the backend response for generation."}