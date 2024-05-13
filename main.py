from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle

app = FastAPI()

with open('stacking_clf.pkl', 'rb') as f:
    stacking_clf = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Titanic Survival Prediction</title>
        </head>
        <body>
            <h1>Welcome to Titanic Survival Prediction</h1>
            <form method="post">
                <label for="pclass">Passenger Class:</label>
                <input type="number" id="pclass" name="pclass" required>
                <label for="sex">Sex:</label>
                <select id="sex" name="sex" required>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                </select>
                <label for="age">Age:</label>
                <input type="number" id="age" name="age" required>
                <label for="sibsp">Number of Siblings/Spouses Aboard:</label>
                <input type="number" id="sibsp" name="sibsp" required>
                <label for="parch">Number of Parents/Children Aboard:</label>
                <input type="number" id="parch" name="parch" required>
                <label for="fare">Fare:</label>
                <input type="number" id="fare" name="fare" required>
                <label for="embarked">Port of Embarkation:</label>
                <select id="embarked" name="embarked" required>
                    <option value="C">Cherbourg</option>
                    <option value="Q">Queenstown</option>
                    <option value="S">Southampton</option>
                </select>
                <button type="submit">Predict Survival</button>
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(pclass: int = Form(...), sex: str = Form(...), age: int = Form(...), sibsp: int = Form(...),
                   parch: int = Form(...), fare: int = Form(...), embarked: str = Form(...)):
    prediction = stacking_clf.predict([[pclass, sex, age, sibsp, parch, fare, embarked]])
    if prediction == 1:
        result = "likely"
    else:
        result = "unlikely"
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Prediction Result</title>
        </head>
        <body>
            <h1>Prediction Result</h1>
            <p>The passenger is <strong>{result}</strong> to have survived.</p>
        </body>
    </html>
    """
