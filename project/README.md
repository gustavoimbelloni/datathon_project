# Datathon - Case Passos Mágicos

Este projeto consiste em uma solução completa de Machine Learning para estimar o **risco de defasagem escolar** de estudantes da Associação Passos Mágicos, incluindo uma API robusta e um Dashboard Web interativo.

## Estrutura do Projeto

```
project/
├── app/                    # API Flask e modelos serializados
│   ├── main.py            # Servidor Flask
│   ├── routes.py          # Rotas da API
│   └── model/             # Modelos treinados (.joblib)
├── src/                   # Pipeline de Machine Learning
│   ├── preprocessing.py   # Limpeza e preparação de dados
│   ├── feature_engineering.py  # Engenharia de atributos
│   ├── train.py           # Treinamento do modelo
│   ├── evaluate.py        # Avaliação de métricas
│   └── utils.py           # Funções auxiliares
├── tests/                 # Testes unitários
│   ├── test_preprocessing.py
│   └── test_model.py
├── dashboard.py           # Interface Streamlit
├── Dockerfile             # Containerização
├── docker-compose.yml     # Orquestração de containers
├── requirements.txt       # Dependências Python
└── README.md             # Este arquivo
```

## Como Executar

### Opção 1: Localmente (Recomendado para Desenvolvimento)

#### Pré-requisitos
- Python 3.8+
- pip (gerenciador de pacotes)

#### Passos

1. **Clone ou extraia o projeto**
   ```bash
   unzip datathon_project.zip
   cd project
   ```

2. **Crie um ambiente virtual** (opcional, mas recomendado)
   ```bash
   python -m venv venv
   # No Windows:
   venv\Scripts\activate
   # No Mac/Linux:
   source venv/bin/activate
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Coloque o arquivo de dados no diretório correto**
   - Copie `BASEDEDADOSPEDE2024-DATATHON.xlsx` para a raiz do projeto
   - Ou atualize o caminho em `src/train.py`

5. **Treine o modelo** (opcional, já vem pré-treinado)
   ```bash
   python src/train.py
   ```

6. **Inicie a API em um terminal**
   ```bash
   python app/main.py
   ```
   A API estará disponível em `http://localhost:5000`

7. **Inicie o Dashboard em outro terminal**
   ```bash
   streamlit run dashboard.py
   ```
   O Dashboard abrirá automaticamente em `http://localhost:8501`

### Opção 2: Com Docker (Recomendado para Produção)

#### Pré-requisitos
- Docker instalado
- Docker Compose (opcional)

#### Passos

**Apenas com Docker:**
```bash
# Construir a imagem
docker build -t datathon-passos-magicos .

# Rodar a API
docker run -p 5000:5000 datathon-passos-magicos python app/main.py

# Em outro terminal, rodar o Dashboard
docker run -p 8501:8501 datathon-passos-magicos streamlit run dashboard.py --server.address 0.0.0.0
```

**Com Docker Compose (Mais fácil):**
```bash
docker-compose up
```

Isso iniciará automaticamente:
- API em `http://localhost:5000`
- Dashboard em `http://localhost:8501`

## Funcionalidades do Dashboard

### 📊 Consulta Individual
- Insira dados de um aluno específico
- Visualize o risco de defasagem em tempo real
- Veja gráficos interativos de probabilidade
- Análise dos indicadores educacionais

### 📈 Análise em Lote
- Carregue um arquivo CSV ou Excel com múltiplos alunos
- Processe todos os registros de uma vez
- Visualize estatísticas agregadas
- Baixe os resultados em CSV

### ℹ️ Sobre
- Informações sobre o projeto
- Descrição dos indicadores utilizados
- Links para mais informações

## API Endpoints

### GET `/health`
Verifica se a API está funcionando.

```bash
curl http://localhost:5000/health
```

**Resposta:**
```json
{"status": "healthy"}
```

### POST `/predict`
Realiza predições de risco de defasagem.

**Exemplo de Chamada:**
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '[{
       "Fase": 1,
       "Idade 22": 10,
       "Gênero": "M",
       "Ano ingresso": 2020,
       "IAA": 8.0,
       "IEG": 9.0,
       "IPS": 7.0,
       "IDA": 8.5,
       "Matem": 8.0,
       "Portug": 7.5,
       "IPV": 8.0,
       "IAN": 9.0
     }]'
```

**Resposta:**
```json
{
  "predictions": [
    {
      "risco_defasagem": false,
      "probabilidade": 0.15
    }
  ]
}
```

## Indicadores Utilizados

| Indicador | Descrição |
|-----------|-----------|
| **IAA** | Indicador de Auto Avaliação - Média das notas de auto avaliação |
| **IEG** | Indicador de Engajamento - Nível de engajamento do aluno |
| **IPS** | Indicador Psicossocial - Aspectos psicossociais do desenvolvimento |
| **IDA** | Indicador de Aprendizagem - Desempenho acadêmico geral |
| **IPV** | Indicador de Ponto de Virada - Indicador de transformação |
| **IAN** | Indicador de Adequação ao Nível - Adequação ao nível escolar |

## Testes

Para rodar os testes unitários com cobertura:

```bash
pytest --cov=src tests/
```

Objetivo: Manter cobertura mínima de **80%** de testes unitários.

## Monitoramento e Drift

A API registra logs de todas as predições. Para monitoramento avançado de drift, considere integrar:
- **Evidently AI**: Detecção automática de drift
- **MLflow**: Rastreamento de experimentos e modelos
- **Prometheus + Grafana**: Métricas e dashboards

## Estrutura de Dados Esperada

O modelo espera os seguintes campos:

```python
{
    "Fase": int,                    # Fase do aluno (1-10)
    "Idade 22": int,                # Idade em 2022
    "Gênero": str,                  # "M" ou "F"
    "Ano ingresso": int,            # Ano que ingressou (2010-2024)
    "IAA": float,                   # 0-10
    "IEG": float,                   # 0-10
    "IPS": float,                   # 0-10
    "IDA": float,                   # 0-10
    "Matem": float,                 # 0-10 (Nota de Matemática)
    "Portug": float,                # 0-10 (Nota de Português)
    "IPV": float,                   # 0-10
    "IAN": float                    # 0-10
}
```

## Troubleshooting

### Erro: "Connection refused" ao conectar à API
- Certifique-se de que a API está rodando (`python app/main.py`)
- Verifique se está usando a URL correta (padrão: `http://localhost:5000`)

### Erro: "ModuleNotFoundError"
- Instale todas as dependências: `pip install -r requirements.txt`
- Certifique-se de estar no ambiente virtual correto

### Erro: "No module named 'openpyxl'"
- Instale a dependência: `pip install openpyxl`

## Contribuições

Este projeto foi desenvolvido para o Datathon da Associação Passos Mágicos com o objetivo de transformar vidas através da educação.

## Licença

Este projeto é fornecido como está para fins educacionais e de pesquisa.

---

**Desenvolvido com ❤️ para transformar vidas através da educação**

Para mais informações sobre a Associação Passos Mágicos, visite: [passosmagicos.org.br](https://www.passosmagicos.org.br/)
