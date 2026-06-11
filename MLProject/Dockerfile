FROM python:3.12.7-slim

WORKDIR /app

COPY MLProject/wine_preprocessing/ /app/wine_preprocessing/
COPY MLProject/modelling.py /app/modelling.py

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn \
    pandas \
    numpy \
    matplotlib

EXPOSE 5002

CMD ["mlflow", "models", "serve", "--host", "0.0.0.0", "--port", "5002"]