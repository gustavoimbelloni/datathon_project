#!/bin/bash

echo "🚀 Iniciando a Central de Inteligência Educacional..."
echo ""
echo "Opções:"
echo "1) Rodar apenas a API (porta 5000)"
echo "2) Rodar apenas o Dashboard Elite (porta 8501)"
echo "3) Rodar ambos (recomendado)"
echo "4) Rodar com Docker Compose"
echo ""
read -p "Escolha uma opção (1-4): " opcao

case $opcao in
    1)
        echo "Iniciando API..."
        python app/main.py
        ;;
    2)
        echo "Iniciando Dashboard Elite..."
        streamlit run dashboard_elite.py
        ;;
    3)
        echo "Iniciando API em background..."
        python app/main.py &
        sleep 2
        echo "Iniciando Dashboard Elite..."
        streamlit run dashboard_elite.py
        ;;
    4)
        echo "Iniciando com Docker Compose..."
        docker-compose up
        ;;
    *)
        echo "Opção inválida!"
        ;;
esac
