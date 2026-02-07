# Guia Rápido - Dashboard Streamlit

## 🚀 Como Iniciar o Dashboard

### Opção 1: Localmente (Mais Rápido)

```bash
# 1. Certifique-se de que a API está rodando
python app/main.py

# 2. Em outro terminal, inicie o Dashboard
streamlit run dashboard.py
```

O Dashboard abrirá automaticamente em `http://localhost:8501`

### Opção 2: Com Docker Compose (Recomendado)

```bash
docker-compose up
```

Isso inicia automaticamente:
- API em `http://localhost:5000`
- Dashboard em `http://localhost:8501`

---

## 📊 Usando o Dashboard

### 1. Consulta Individual

**Passo a passo:**
1. Selecione "📊 Consulta Individual" no menu lateral
2. Preencha os dados do aluno:
   - **Fase**: Nível escolar (1-10)
   - **Idade**: Idade atual
   - **Gênero**: Masculino (M) ou Feminino (F)
   - **Ano de Ingresso**: Quando entrou na Passos Mágicos
   - **Indicadores**: Use os sliders para ajustar IAA, IEG, IPS, IDA, IPV, IAN
   - **Notas**: Matemática e Português (0-10)

3. Clique no botão "🔍 Analisar Risco"

**Resultado:**
- ✅ **RISCO BAIXO** (verde): Aluno está bem encaminhado
- ⚠️ **RISCO ALTO** (vermelho): Necessário acompanhamento especial
- Gráfico visual com probabilidade
- Resumo dos indicadores

---

### 2. Análise em Lote

**Passo a passo:**
1. Selecione "📈 Análise em Lote" no menu lateral
2. Clique em "Carregue um arquivo CSV ou Excel"
3. Escolha um arquivo com múltiplos alunos
4. Clique em "🔍 Analisar Todos os Registros"

**Resultado:**
- Estatísticas gerais (Total, Alto Risco, Baixo Risco)
- Gráfico de pizza com distribuição de risco
- Tabela interativa com resultados
- Botão para baixar resultados em CSV

**Formato esperado do arquivo:**
```
Fase,Idade 22,Gênero,Ano ingresso,IAA,IEG,IPS,IDA,Matem,Portug,IPV,IAN
1,10,M,2020,8.0,9.0,7.0,8.5,8.0,7.5,8.0,9.0
2,11,F,2019,7.5,8.0,8.5,7.0,7.0,8.0,7.5,8.0
```

---

### 3. Sobre

Informações sobre o projeto, indicadores utilizados e links úteis.

---

## 🎨 Entendendo as Cores

| Cor | Significado | Ação Recomendada |
|-----|------------|------------------|
| 🟢 Verde | Risco Baixo | Continuar acompanhamento regular |
| 🟡 Amarelo | Risco Médio | Intensificar acompanhamento |
| 🔴 Vermelho | Risco Alto | Intervenção imediata necessária |

---

## 📈 Interpretando os Indicadores

- **IAA (Auto Avaliação)**: Como o aluno se avalia
- **IEG (Engajamento)**: Nível de participação e interesse
- **IPS (Psicossocial)**: Bem-estar emocional e social
- **IDA (Aprendizagem)**: Desempenho acadêmico
- **IPV (Ponto de Virada)**: Indicador de transformação
- **IAN (Adequação ao Nível)**: Alinhamento com o nível escolar

---

## 🔧 Troubleshooting

### Dashboard não abre
```bash
# Verifique se Streamlit está instalado
pip install streamlit

# Tente rodar novamente
streamlit run dashboard.py
```

### Erro "Connection refused"
- Certifique-se de que a API está rodando em `http://localhost:5000`
- No menu lateral, você pode alterar a URL da API se necessário

### Arquivo não carrega
- Verifique se o arquivo está em formato CSV ou XLSX
- Certifique-se de que tem as colunas esperadas

---

## 💡 Dicas Profissionais

1. **Para Apresentações**: Use a Análise em Lote para mostrar resultados de um grupo inteiro
2. **Para Decisões Rápidas**: Use a Consulta Individual para avaliar um aluno específico
3. **Para Relatórios**: Baixe os resultados em CSV e importe em ferramentas de análise

---

## 📞 Suporte

Se encontrar problemas:
1. Verifique se a API está rodando
2. Verifique se as dependências estão instaladas
3. Consulte o README.md para mais informações

---

**Desenvolvido com ❤️ para a Associação Passos Mágicos**
