# Używamy oficjalnego obrazu Python 3.9
FROM python:3.9-slim

# Ustawiamy zmienne środowiskowe
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ustawiamy katalog roboczy w kontenerze
WORKDIR /app

# Kopiujemy pliki projektu
COPY . /app/

# Instalujemy zależności
RUN pip install --no-cache-dir -r requirements.txt

# Ustawiamy port dla dashboardu
EXPOSE 8050

# Komenda uruchamiająca aplikację
CMD ["python", "main.py"] 