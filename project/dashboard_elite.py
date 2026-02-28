import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr

# Plotly Express usa statsmodels para trendlines do tipo OLS.
# Mantemos o app resiliente caso a dependência não esteja instalada no ambiente.
try:
    import statsmodels.api as sm  # noqa: F401
    _HAS_STATSMODELS = True
except ModuleNotFoundError:
    _HAS_STATSMODELS = False

# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos - Central de Inteligência Educacional",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
    }
    .risk-medium {
        background-color: #ffffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffaa00;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00aa00;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Configuração da API
API_URL = st.sidebar.text_input("URL da API", value="http://localhost:5000")

st.markdown('<div class="header-title">✨ Central de Inteligência Educacional</div>', unsafe_allow_html=True)
st.markdown("### Associação Passos Mágicos - Sistema de Análise de Risco de Defasagem")
st.markdown("---")

# Menu principal
menu = st.sidebar.radio(
    "📋 Menu Principal",
    ["🏠 Dashboard", "🔍 Explorador de Dados", "📊 Análises Avançadas", 
     "🎯 Simulador de Impacto", "📈 Consulta Individual", "📥 Análise em Lote", "ℹ️ Sobre"]
)

# ============= DASHBOARD (Home) =============
if menu == "🏠 Dashboard":
    st.subheader("Painel de Visão Geral")
    
    # Carregar dados para análise
    try:
        df = pd.read_excel('data\BASE DE DADOS PEDE 2024 - DATATHON.xlsx')
        
        # Calcular KPIs
        total_alunos = len(df)
        inde_medio = df['INDE 22'].mean()
        alunos_em_risco = len(df[df['Defas'] < 0])
        percentual_risco = (alunos_em_risco / total_alunos) * 100
        ponto_virada = len(df[df['Atingiu PV'] == 'Sim'])
        percentual_pv = (ponto_virada / total_alunos) * 100
        
        # Exibir KPIs em cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_alunos}</div>
                <div class="metric-label">Total de Alunos</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{inde_medio:.2f}</div>
                <div class="metric-label">INDE Médio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{percentual_risco:.1f}%</div>
                <div class="metric-label">Em Risco</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{percentual_pv:.1f}%</div>
                <div class="metric-label">Ponto de Virada</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição por Pedra
            pedra_counts = df['Pedra 22'].value_counts()
            fig_pedra = px.pie(
                values=pedra_counts.values,
                names=pedra_counts.index,
                title="Distribuição por Pedra (Classificação)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pedra, use_container_width=True)
        
        with col2:
            # Distribuição de Defasagem
            defas_counts = df['Defas'].value_counts().sort_index()
            fig_defas = px.bar(
                x=defas_counts.index,
                y=defas_counts.values,
                title="Distribuição de Defasagem",
                labels={"x": "Nível de Defasagem", "y": "Quantidade de Alunos"},
                color=defas_counts.values,
                color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig_defas, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Notas por Disciplina
            fig_notas = go.Figure()
            fig_notas.add_trace(go.Box(y=df['Matem'], name='Matemática', marker_color='#1f77b4'))
            fig_notas.add_trace(go.Box(y=df['Portug'], name='Português', marker_color='#ff7f0e'))
            fig_notas.update_layout(title="Distribuição de Notas", yaxis_title="Nota")
            st.plotly_chart(fig_notas, use_container_width=True)
        
        with col4:
            # Indicadores Médios
            indicadores = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN']
            medias = [df[ind].mean() for ind in indicadores]
            fig_ind = go.Figure(data=go.Scatterpolar(
                r=medias,
                theta=indicadores,
                fill='toself',
                name='Média de Indicadores'
            ))
            fig_ind.update_layout(title="Perfil Médio de Indicadores")
            st.plotly_chart(fig_ind, use_container_width=True)
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")

# ============= EXPLORADOR DE DADOS =============
elif menu == "🔍 Explorador de Dados":
    st.subheader("Explorador Avançado de Dados")
    
    try:
        df = pd.read_excel('data\BASE DE DADOS PEDE 2024 - DATATHON.xlsx')
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fase_filter = st.multiselect("Fase", sorted(df['Fase'].unique()), default=sorted(df['Fase'].unique()))
        
        with col2:
            genero_filter = st.multiselect("Gênero", df['Gênero'].unique(), default=df['Gênero'].unique())
        
        with col3:
            ano_filter = st.multiselect("Ano de Ingresso", sorted(df['Ano ingresso'].unique()), default=sorted(df['Ano ingresso'].unique()))
        
        # Aplicar filtros
        df_filtered = df[
            (df['Fase'].isin(fase_filter)) &
            (df['Gênero'].isin(genero_filter)) &
            (df['Ano ingresso'].isin(ano_filter))
        ]
        
        # Opções de ordenação
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("Ordenar por:", 
                                   ["INDE 22", "Defas", "Matem", "Portug", "IAA", "IEG", "IPS", "IDA"])
        with col2:
            sort_order = st.radio("Ordem:", ["Crescente", "Decrescente"])
        
        ascending = sort_order == "Crescente"
        df_filtered = df_filtered.sort_values(by=sort_by, ascending=ascending)
        
        # Exibir tabela
        st.dataframe(df_filtered[['RA', 'Fase', 'Gênero', 'Pedra 22', 'INDE 22', 'Defas', 'Matem', 'Portug', 'IAA', 'IEG']], 
                    use_container_width=True)
        
        st.info(f"Total de registros: {len(df_filtered)}")
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")

# ============= ANÁLISES AVANÇADAS =============
elif menu == "📊 Análises Avançadas":
    st.subheader("Análises Avançadas e Correlações")
    
    try:
        df = pd.read_excel('data\BASE DE DADOS PEDE 2024 - DATATHON.xlsx')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlação entre Engajamento e Defasagem
            valid_data = df[['IEG', 'Defas']].dropna()
            if len(valid_data) > 0:
                corr_ieg = pearsonr(valid_data['IEG'], valid_data['Defas'])[0]
                fig_ieg = px.scatter(
                    valid_data,
                    x='IEG',
                    y='Defas',
                    trendline='ols' if _HAS_STATSMODELS else None,
                    title=f"Engajamento vs Defasagem (Correlação: {corr_ieg:.3f})",
                    labels={"IEG": "Indicador de Engajamento", "Defas": "Defasagem"}
                )
                st.plotly_chart(fig_ieg, use_container_width=True)
                if not _HAS_STATSMODELS:
                    st.info("Trendline (OLS) indisponível: instale `statsmodels` para exibir a linha de tendência.")
        
        with col2:
            # Correlação entre Auto Avaliação e Desempenho
            valid_data = df[['IAA', 'Matem']].dropna()
            if len(valid_data) > 0:
                corr_iaa = pearsonr(valid_data['IAA'], valid_data['Matem'])[0]
                fig_iaa = px.scatter(
                    valid_data,
                    x='IAA',
                    y='Matem',
                    trendline='ols' if _HAS_STATSMODELS else None,
                    title=f"Auto Avaliação vs Matemática (Correlação: {corr_iaa:.3f})",
                    labels={"IAA": "Indicador de Auto Avaliação", "Matem": "Nota de Matemática"}
                )
                st.plotly_chart(fig_iaa, use_container_width=True)
                if not _HAS_STATSMODELS:
                    st.info("Trendline (OLS) indisponível: instale `statsmodels` para exibir a linha de tendência.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Mapa de Calor: Notas por Fase
            pivot_notas = df.groupby('Fase')[['Matem', 'Portug']].mean()
            fig_heatmap = px.imshow(
                pivot_notas.T,
                labels=dict(x="Fase", y="Disciplina", color="Nota Média"),
                title="Mapa de Calor: Notas Médias por Fase",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col4:
            # Evolução de Indicadores por Fase
            indicadores_por_fase = df.groupby('Fase')[['IAA', 'IEG', 'IPS', 'IDA']].mean()
            fig_evolucao = px.line(
                indicadores_por_fase,
                title="Evolução de Indicadores por Fase",
                markers=True
            )
            st.plotly_chart(fig_evolucao, use_container_width=True)
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")

# ============= SIMULADOR DE IMPACTO =============
elif menu == "🎯 Simulador de Impacto":
    st.subheader("Simulador de Impacto (What-If)")
    
    st.info("Ajuste os indicadores de um aluno fictício e veja como o risco de defasagem mudaria com melhorias específicas.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cenário Base")
        fase_sim = st.number_input("Fase", min_value=1, max_value=10, value=1, key="sim_fase")
        idade_sim = st.number_input("Idade", min_value=5, max_value=25, value=10, key="sim_idade")
        genero_sim = st.selectbox("Gênero", ["M", "F"], key="sim_genero")
        ano_ingresso_sim = st.number_input("Ano de Ingresso", min_value=2010, max_value=2024, value=2020, key="sim_ano")
    
    with col2:
        st.markdown("### Indicadores Atuais")
        iaa_sim = st.slider("IAA (Auto Avaliação)", 0.0, 10.0, 5.0, key="sim_iaa")
        ieg_sim = st.slider("IEG (Engajamento)", 0.0, 10.0, 5.0, key="sim_ieg")
        ips_sim = st.slider("IPS (Psicossocial)", 0.0, 10.0, 5.0, key="sim_ips")
        ida_sim = st.slider("IDA (Aprendizagem)", 0.0, 10.0, 5.0, key="sim_ida")
    
    col3, col4 = st.columns(2)
    
    with col3:
        matem_sim = st.number_input("Nota de Matemática", 0.0, 10.0, 5.0, key="sim_matem")
        portug_sim = st.number_input("Nota de Português", 0.0, 10.0, 5.0, key="sim_portug")
    
    with col4:
        ipv_sim = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 5.0, key="sim_ipv")
        ian_sim = st.slider("IAN (Adequação ao Nível)", 0.0, 10.0, 5.0, key="sim_ian")
    
    # Fazer predição do cenário base
    if st.button("🔍 Analisar Cenário Base"):
        try:
            data_base = {
                "Fase": fase_sim,
                "Idade 22": idade_sim,
                "Gênero": genero_sim,
                "Ano ingresso": ano_ingresso_sim,
                "IAA": iaa_sim,
                "IEG": ieg_sim,
                "IPS": ips_sim,
                "IDA": ida_sim,
                "Matem": matem_sim,
                "Portug": portug_sim,
                "IPV": ipv_sim,
                "IAN": ian_sim
            }
            
            response = requests.post(f"{API_URL}/predict", json=[data_base])
            
            if response.status_code == 200:
                result = response.json()
                pred_base = result["predictions"][0]
                prob_base = pred_base["probabilidade"]
                
                st.markdown("---")
                st.subheader("Simulações de Impacto")
                
                # Simulações
                cenarios = {
                    "📈 +1.0 em Engajamento": {"IEG": ieg_sim + 1.0},
                    "📚 +1.0 em Aprendizagem": {"IDA": ida_sim + 1.0},
                    "💪 +1.0 em Psicossocial": {"IPS": ips_sim + 1.0},
                    "🎯 +1.0 em Ponto de Virada": {"IPV": ipv_sim + 1.0},
                    "🌟 +1.0 em Todos os Indicadores": {
                        "IEG": ieg_sim + 1.0,
                        "IDA": ida_sim + 1.0,
                        "IPS": ips_sim + 1.0,
                        "IPV": ipv_sim + 1.0,
                        "IAA": iaa_sim + 1.0,
                        "IAN": ian_sim + 1.0
                    }
                }
                
                resultados_sim = []
                
                for cenario_nome, alteracoes in cenarios.items():
                    data_sim = data_base.copy()
                    data_sim.update(alteracoes)
                    
                    # Limitar valores entre 0 e 10
                    for key in data_sim:
                        if isinstance(data_sim[key], float) and key not in ["Fase", "Idade 22", "Ano ingresso"]:
                            data_sim[key] = min(10.0, max(0.0, data_sim[key]))
                    
                    response_sim = requests.post(f"{API_URL}/predict", json=[data_sim])
                    
                    if response_sim.status_code == 200:
                        result_sim = response_sim.json()
                        pred_sim = result_sim["predictions"][0]
                        prob_sim = pred_sim["probabilidade"]
                        reducao = ((prob_base - prob_sim) / prob_base * 100) if prob_base > 0 else 0
                        
                        resultados_sim.append({
                            "Cenário": cenario_nome,
                            "Risco Base": f"{prob_base*100:.1f}%",
                            "Risco Simulado": f"{prob_sim*100:.1f}%",
                            "Redução": f"{reducao:.1f}%"
                        })
                
                df_sim = pd.DataFrame(resultados_sim)
                st.dataframe(df_sim, use_container_width=True)
                
                # Gráfico comparativo
                probs_base = [prob_base] * len(cenarios)
                probs_sim = []
                
                for cenario_nome, alteracoes in cenarios.items():
                    data_sim = data_base.copy()
                    data_sim.update(alteracoes)
                    for key in data_sim:
                        if isinstance(data_sim[key], float) and key not in ["Fase", "Idade 22", "Ano ingresso"]:
                            data_sim[key] = min(10.0, max(0.0, data_sim[key]))
                    response_sim = requests.post(f"{API_URL}/predict", json=[data_sim])
                    if response_sim.status_code == 200:
                        result_sim = response_sim.json()
                        probs_sim.append(result_sim["predictions"][0]["probabilidade"])
                
                fig_comparacao = go.Figure(data=[
                    go.Bar(name='Risco Base', x=list(cenarios.keys()), y=[p*100 for p in probs_base], marker_color='#ff6b6b'),
                    go.Bar(name='Risco Simulado', x=list(cenarios.keys()), y=[p*100 for p in probs_sim], marker_color='#51cf66')
                ])
                fig_comparacao.update_layout(
                    title="Comparação de Risco: Cenário Base vs Simulações",
                    yaxis_title="Risco (%)",
                    barmode='group',
                    height=500
                )
                st.plotly_chart(fig_comparacao, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erro ao processar simulação: {str(e)}")

# ============= CONSULTA INDIVIDUAL =============
elif menu == "📈 Consulta Individual":
    st.subheader("Consulta Individual de Risco")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fase = st.number_input("Fase", min_value=1, max_value=10, value=1)
        idade = st.number_input("Idade", min_value=5, max_value=25, value=10)
        genero = st.selectbox("Gênero", ["M", "F"])
        ano_ingresso = st.number_input("Ano de Ingresso", min_value=2010, max_value=2024, value=2020)
    
    with col2:
        iaa = st.slider("IAA (Auto Avaliação)", 0.0, 10.0, 8.0)
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 8.0)
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 8.0)
        ida = st.slider("IDA (Aprendizagem)", 0.0, 10.0, 8.0)
    
    col3, col4 = st.columns(2)
    
    with col3:
        matem = st.number_input("Nota de Matemática", 0.0, 10.0, 7.5)
        portug = st.number_input("Nota de Português", 0.0, 10.0, 7.5)
    
    with col4:
        ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 8.0)
        ian = st.slider("IAN (Adequação ao Nível)", 0.0, 10.0, 8.0)
    
    if st.button("🔍 Analisar Risco", key="analyze_individual"):
        try:
            data = {
                "Fase": fase,
                "Idade 22": idade,
                "Gênero": genero,
                "Ano ingresso": ano_ingresso,
                "IAA": iaa,
                "IEG": ieg,
                "IPS": ips,
                "IDA": ida,
                "Matem": matem,
                "Portug": portug,
                "IPV": ipv,
                "IAN": ian
            }
            
            response = requests.post(f"{API_URL}/predict", json=[data])
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["predictions"][0]
                
                risco = prediction["risco_defasagem"]
                probabilidade = prediction["probabilidade"]
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if risco:
                        st.markdown(f"""
                        <div class="risk-high">
                            <h3>⚠️ RISCO ALTO</h3>
                            <p>Probabilidade: <strong>{probabilidade*100:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-low">
                            <h3>✅ RISCO BAIXO</h3>
                            <p>Probabilidade: <strong>{(1-probabilidade)*100:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=probabilidade * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risco (%)"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 33], 'color': "#90EE90"},
                                {'range': [33, 66], 'color': "#FFD700"},
                                {'range': [66, 100], 'color': "#FF6B6B"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    st.metric("Média de Indicadores", f"{(iaa + ieg + ips + ida + ipv + ian) / 6:.2f}")
                    st.metric("Média de Notas", f"{(matem + portug) / 2:.2f}")
                    st.metric("Tempo na Instituição", f"{2024 - ano_ingresso} anos")
        
        except Exception as e:
            st.error(f"Erro ao conectar à API: {str(e)}")

# ============= ANÁLISE EM LOTE =============
elif menu == "📥 Análise em Lote":
    st.subheader("Análise em Lote")
    
    uploaded_file = st.file_uploader("Carregue um arquivo CSV ou Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write(f"Arquivo carregado com {len(df)} registros")
            
            if st.button("🔍 Analisar Todos os Registros"):
                predictions = []
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    try:
                        data = row.to_dict()
                        response = requests.post(f"{API_URL}/predict", json=[data])
                        
                        if response.status_code == 200:
                            result = response.json()
                            pred = result["predictions"][0]
                            predictions.append({
                                "Índice": idx,
                                "Risco": "Alto" if pred["risco_defasagem"] else "Baixo",
                                "Probabilidade": f"{pred['probabilidade']*100:.1f}%"
                            })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    except:
                        pass
                
                results_df = pd.DataFrame(predictions)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Analisado", len(results_df))
                with col2:
                    alto_risco = len(results_df[results_df["Risco"] == "Alto"])
                    st.metric("Alto Risco", alto_risco)
                with col3:
                    baixo_risco = len(results_df[results_df["Risco"] == "Baixo"])
                    st.metric("Baixo Risco", baixo_risco)
                
                fig = px.pie(results_df, names="Risco", title="Distribuição de Risco")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(results_df, use_container_width=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Baixar Resultados (CSV)",
                    data=csv,
                    file_name=f"analise_risco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

# ============= SOBRE =============
elif menu == "ℹ️ Sobre":
    st.subheader("Sobre o Projeto")
    
    st.markdown("""
    ### 🌟 Datathon - Associação Passos Mágicos
    
    #### Missão
    Transformar a vida de crianças e jovens de baixa renda através da educação, identificando 
    precocemente aqueles em risco de defasagem escolar para intervenção personalizada.
    
    #### Tecnologia Utilizada
    - **Backend**: Python, Flask, Scikit-Learn
    - **Frontend**: Streamlit
    - **Modelo**: Random Forest Classifier
    - **Containerização**: Docker & Docker Compose
    - **Visualização**: Plotly
    
    #### Indicadores Educacionais
    
    | Indicador | Descrição |
    |-----------|-----------|
    | **IAA** | Indicador de Auto Avaliação |
    | **IEG** | Indicador de Engajamento |
    | **IPS** | Indicador Psicossocial |
    | **IDA** | Indicador de Aprendizagem |
    | **IPV** | Indicador de Ponto de Virada |
    | **IAN** | Indicador de Adequação ao Nível |
    
    #### Funcionalidades
    - 📊 **Dashboard**: Visão geral com KPIs de impacto
    - 🔍 **Explorador de Dados**: Filtros avançados e análise exploratória
    - 📈 **Análises Avançadas**: Correlações e tendências
    - 🎯 **Simulador de Impacto**: Cenários What-If
    - 📈 **Consulta Individual**: Predição de risco por aluno
    - 📥 **Análise em Lote**: Processamento de múltiplos alunos
    
    #### Sobre a Associação
    A Associação Passos Mágicos tem uma trajetória de 32 anos de atuação transformando 
    a vida de crianças e jovens de baixa renda através da educação de qualidade.
    
    📍 **Endereço**: Rua Francisco Volante, 13 - Jardim Brasil - Embu-Guaçu - SP
    🌐 **Site**: [passosmagicos.org.br](https://www.passosmagicos.org.br/)
    """)
    
    st.markdown("---")
    st.markdown("Desenvolvido com ❤️ para transformar vidas através da educação")
