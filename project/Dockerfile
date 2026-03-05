FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de requisitos e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código do projeto
COPY . /app/

# Portas: API (5001) e Streamlit (8502) - cada serviço usa uma no compose
EXPOSE 5001 8502

# Comando padrão (sobrescrito pelo docker-compose por serviço)
CMD ["python", "app/main.py"]
