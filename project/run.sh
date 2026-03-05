#!/bin/bash
# run.sh - Sobe API, Streamlit e executa complete_pipeline_test.py gerando logs
# Uso: ./run.sh [docker|local]  (default: docker)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="${LOG_DIR:-./logs}"
LOG_FILE="${LOG_DIR}/pipeline_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

MODE="${1:-docker}"

wait_for_api() {
  local url="${1:-http://localhost:5001/health}"
  local max_attempts=30
  local attempt=1
  echo "Aguardando API em $url ..."
  while [ $attempt -le $max_attempts ]; do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo "API disponível."
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done
  echo "AVISO: API não respondeu a tempo. Continuando mesmo assim."
  return 1
}

run_docker() {
  if command -v docker-compose >/dev/null 2>&1; then
    DCOMPOSE="docker-compose"
  else
    DCOMPOSE="docker compose"
  fi
  echo "=============================================="
  echo "Iniciando com Docker Compose"
  echo "=============================================="
  $DCOMPOSE up -d --build
  wait_for_api "http://localhost:5001/health" || true
  echo ""
  echo "Executando complete_pipeline_test.py e gerando logs..."
  $DCOMPOSE run --rm api python complete_pipeline_test.py 2>&1 | tee "$LOG_FILE"
  echo ""
  echo "Logs salvos em: $LOG_FILE"
  echo "Serviços continuam rodando. Para parar: $DCOMPOSE down"
}

run_local() {
  echo "=============================================="
  echo "Iniciando localmente (API + Streamlit + Testes)"
  echo "=============================================="
  # 1) API em background
  echo "1) Subindo API (porta 5001)..."
  python app/main.py &
  API_PID=$!
  sleep 2
  wait_for_api "http://localhost:5001/health" || true

  # 2) Streamlit em background
  echo "2) Subindo Streamlit (porta 8502)..."
  streamlit run dashboard_elite.py --server.port 8502 --server.address 0.0.0.0 &
  STREAMLIT_PID=$!
  sleep 3

  # 3) Pipeline test com logs
  echo "3) Executando complete_pipeline_test.py e gerando logs..."
  trap "kill $API_PID $STREAMLIT_PID 2>/dev/null; exit" INT TERM
  python complete_pipeline_test.py 2>&1 | tee "$LOG_FILE"
  echo ""
  echo "Logs salvos em: $LOG_FILE"
  echo "API (PID $API_PID) e Streamlit (PID $STREAMLIT_PID) ainda rodando. Para parar: kill $API_PID $STREAMLIT_PID"
}

case "$MODE" in
  docker)
    run_docker
    ;;
  local)
    run_local
    ;;
  *)
    echo "Uso: $0 [docker|local]"
    echo "  docker - sobe API e Streamlit com Docker Compose e roda os testes no container (default)"
    echo "  local  - sobe API e Streamlit em background no host e roda os testes localmente"
    exit 1
    ;;
esac
