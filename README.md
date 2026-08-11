# Datathon - Plataforma de Experimentação Adaptativa de Ofertas

Solução end-to-end de Machine Learning Engineering para decidir, de forma adaptativa, **qual canal/oferta apresentar a cada cliente elegível** usando *multi-armed bandits* (Thompson Sampling e Epsilon-Greedy) comparados a um baseline determinístico.

## Visão do problema

Uma instituição financeira digital precisa escolher, em canais digitais, a melhor abordagem para cada cliente. Regras fixas desperdiçam tráfego e testes A/B longos demoram a reagir. A solução formula cada contato de campanha como uma decisão de bandit:

- **Braços (ações)**: canal/abordagem de contato - `cellular` vs. `telephone`.
- **Recompensa**: conversão observada (assinatura do depósito a prazo, `y = yes`).
- **Contexto**: segmento do cliente (faixa etária × posse de crédito), sem atributos sensíveis.
- **Avaliação**: *offline replay* (Li et al., 2011) sobre o log histórico. Como a política de coleta não foi aleatória e suas propensões são desconhecidas, os resultados são comparativos e sujeitos a viés de seleção; não representam uma estimativa causal de uplift.

### Resultados (replay offline, 41.176 eventos após limpeza)

| Política | Conversão | Eventos casados | Taxa de casamento | Uplift vs. baseline |
|---|---:|---:|---:|---:|
| Melhor braço histórico (referência retrospectiva) | 14,74% | 26.135 | 63,47% | +181,7% |
| **Thompson Sampling** | **14,68%** | **26.054** | **63,27%** | **+180,7%** |
| Epsilon-Greedy (`epsilon=0,1`) | 14,45% | 25.508 | 61,95% | +176,2% |
| Thompson Sampling contextual | 14,05% | 25.475 | 61,87% | +168,4% |
| Baseline - regra fixa (`telephone`) | 5,23% | 15.041 | 36,53% | - |

O Thompson Sampling (prior `Beta(1,1)`, uniforme e não-informativa - documentada em [src/bandits.py](src/bandits.py)) aprende o melhor braço sem conhecê-lo de antemão e praticamente empata com a referência retrospectiva. O uplift observado não deve ser interpretado como efeito causal, pois canal e período da campanha podem estar confundidos no log.

## Base de dados (Kaggle)

- **Link**: <https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing>
- **Arquivo**: `bank-additional-full.csv` (41.188 linhas, 20 features + alvo) - mesma distribuição do repositório UCI Bank Marketing ("additional"), baixada automaticamente pelo pipeline.
- **Licença/fonte**: UCI Machine Learning Repository (CC BY 4.0); Moro, Cortez & Rita (2014).
- **Target**: `y` (assinatura de depósito a prazo) → `converted`.
- **Vazamento temporal**: a coluna `duration` é **descartada** após a remoção das 12 duplicatas reais (só é conhecida após a ligação, conforme a documentação da própria base).
- **Versionamento**: cada execução gera `data/processed/data_manifest.json` com URL de origem, SHA-256 do arquivo bruto, dimensões e colunas processadas. Dados brutos e CSV processado não são versionados no Git.
- **Limitações**: log não-aleatorizado (viés da política de coleta), apenas 2 braços reais, campanhas de 2008–2010 (não-estacionariedade).

## Estrutura do repositório

```
├── notebooks/
│   └── 01_eda_e_bandits.ipynb   # Etapas 1-4: EDA, preparação, baseline vs. bandit, golden set
├── src/
│   ├── data_prep.py             # Download, limpeza, features e segmentos
│   ├── bandits.py               # Baseline, Epsilon-Greedy, Thompson Sampling + replay evaluation
│   ├── train.py                 # Comparação de políticas com tracking no MLflow
│   └── api.py                   # Etapa 5: serviço FastAPI de recomendação
├── models/                      # Estados (posteriores) das políticas treinadas
├── data/                        # raw/ e processed/ (gerados pelo pipeline)
├── requirements.txt
├── tests/                       # Testes de dados, políticas, serialização e API
└── README.md
```

## Como executar localmente

Pré-requisito: Python 3.11+.

```bash
# 1. Ambiente
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/mac: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Pipeline de dados (download + preparação)
python -m src.data_prep

# 3. Experimentos: baseline vs. bandits + registro no MLflow
python -m src.train

# 4. MLflow UI (parâmetros, métricas e artefatos dos runs)
mlflow ui --backend-store-uri sqlite:///mlflow.db
# abrir http://127.0.0.1:5000

# 5. Serviço de recomendação (FastAPI)
uvicorn src.api:app --reload
# abrir http://127.0.0.1:8000/docs

# 6. Testes automatizados
pytest -q
```

Exemplo de chamada à API:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"age": 67, "job": "retired", "housing": "no", "loan": "no"}'
```

Resposta: braço recomendado, segmento, conversão esperada e posteriores Beta por braço (transparência da decisão).

O notebook das Etapas 1–4 pode ser aberto direto no Jupyter/VS Code: [notebooks/01_eda_e_bandits.ipynb](notebooks/01_eda_e_bandits.ipynb) (já executado, com outputs salvos).

## Golden Set - 5 casos de teste (Etapa 4)

| Cliente | Segmento | Recomendação | Conversão esperada | Faz sentido? |
|---|---|---|---|---|
| C1 - jovem estudante sem crédito | jovem_sem_credito | cellular | 21,1% | ✔ melhor braço do segmento |
| C2 - adulto técnico com financiamento | adulto_com_credito | cellular | 12,2% | ✔ melhor braço do segmento |
| C3 - meia-idade gerente com empréstimo | meia_idade_com_credito | cellular | 13,5% | ✔ melhor braço do segmento |
| C4 - sênior aposentado sem crédito | senior_sem_credito | cellular | 47,1% | ✔ melhor braço do segmento |
| C5 - adulto operário com ambos créditos | adulto_com_credito | cellular | 12,2% | ✔ melhor braço do segmento |

## Arquitetura-alvo em nuvem (Etapa 6 - AWS)

Em produção na AWS, o serviço de recomendação (FastAPI) rodaria em contêiner no **ECS Fargate** atrás de um **Application Load Balancer** com **API Gateway**, com a imagem versionada no **ECR**. Os eventos de decisão e recompensa (impressão → conversão) seriam publicados no **Kinesis** e persistidos no **S3** (data lake versionado), enquanto o estado das posteriores dos bandits ficaria no **DynamoDB** para atualização online de baixa latência. O tracking de experimentos usaria **MLflow em EC2/ECS com backend RDS** e artefatos no S3.

A observabilidade seria feita com **CloudWatch** (logs, métricas de conversão por braço/segmento e alarmes de drift ou queda de recompensa), e a governança com **IAM** (menor privilégio), auditoria via **CloudTrail** e re-treino/reavaliação agendados via **Step Functions**. Decisões sensíveis passariam por fila de revisão humana (human-in-the-loop) antes da execução.

## Ciclo de vida MLOps (Etapa 7)

`python -m src.train` registra no **MLflow** (backend SQLite local, `mlflow.db`), para cada política:

- **Parâmetros**: tipo de política, braços, `epsilon`, priors (`alpha`, `beta`), seed, dataset e nº de eventos.
- **Métricas**: taxa de conversão no replay, taxa de casamento, eventos casados e conversões.
- **Artefatos**: estado JSON das posteriores de cada política (usado pela API - o mesmo artefato treinado é o que serve).

## Governança e uso responsável de dados

- **Base legal/finalidade**: dados públicos e anonimizados (UCI/Kaggle) usados exclusivamente para fins educacionais; nenhum dado real de cliente, identificador, renda, gênero ou raça é utilizado.
- **Minimização**: o contexto usa apenas faixa etária e posse de produtos de crédito; colunas não usadas não saem do pipeline local.
- **Retenção**: dados brutos e CSV processado ficam locais (ignorados no Git) e podem ser regenerados; somente o manifesto de versão é mantido no repositório.
- **Humano no loop**: toda resposta da API sinaliza que decisões sensíveis exigem revisão humana antes da execução.
- **Limitações**: ver seção da base de dados e conclusões do notebook.
