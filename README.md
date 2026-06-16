# Email Spam Classifier

Container-ready Flask + TensorFlow email classifier.

## Project Structure

```text
project/
├── backend/
│   ├── app.py
│   ├── predict.py
│   ├── templates/
│   └── static/
├── model/
│   ├── email_classifier.keras
│   ├── labels.json
│   └── metrics.json
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python backend\app.py
```

Open http://localhost:5000.

## Run With Docker

```powershell
docker build -t email-spam-classifier .
docker run --rm -p 5000:5000 email-spam-classifier
```

Open http://localhost:5000.

## Train Model

```powershell
python model\train.py
```
