# Datathon - Plataforma de Experimentação Adaptativa de Ofertas

Solução end-to-end de Machine Learning Engineering para decidir, de forma adaptativa, **qual canal/oferta apresentar a cada cliente elegível** usando *multi-armed bandits* (Thompson Sampling e Epsilon-Greedy) comparados a um baseline determinístico.

## Visão do problema

Uma instituição financeira digital precisa escolher, em canais digitais, a melhor abordagem para cada cliente. Regras fixas desperdiçam tráfego e testes A/B longos demoram a reagir. A solução formula cada contato de campanha como uma decisão de bandit:

- **Braços (ações)**: canal/abordagem de contato - `cellular` vs. `telephone`.
- **Recompensa**: conversão observada (assinatura do depósito a prazo, `y = yes`).
- **Contexto**: segmento do cliente (faixa etária × posse de crédito), sem atributos sensíveis.
- **Avaliação**: *offline replay* (Li et al., 2011) sobre o log histórico. Como a política de coleta não foi aleatória e suas propensões são desconhecidas, os resultados são comparativos e sujeitos a viés de seleção; não representam uma estimativa causal de uplift.

### Resultados (replay offline com validação cruzada de 5 folds, 41.176 eventos)

| Política | Conversão (média ± desvio) | Eventos casados por fold | Taxa de casamento (média ± desvio) | Uplift vs. baseline |
|---|---:|---:|---:|---:|
| Melhor braço histórico (referência retrospectiva) | 14,74% ± 0,41% | 5.227,0 | 63,47% ± 0,44% | +181,7% |
| **Thompson Sampling** | **14,74% ± 0,41%** | **5.227,0** | **63,47% ± 0,44%** | **+181,7%** |
| Epsilon-Greedy (`epsilon=0,1`) | 14,49% ± 0,44% | 5.121,2 | 62,19% ± 0,49% | +177,0% |
| Thompson Sampling contextual | 13,77% ± 0,72% | 4.804,6 | 58,34% ± 4,59% | +163,2% |
| Baseline - regra fixa (`telephone`) | 5,23% ± 0,28% | 3.008,2 | 36,53% ± 0,44% | - |

O Thompson Sampling (prior `Beta(1,1)`, uniforme e não-informativa - documentada em [src/bandits.py](src/bandits.py)) aprende o melhor braço sem conhecê-lo de antemão e praticamente empata com a referência retrospectiva. O uplift observado não deve ser interpretado como efeito causal, pois canal e período da campanha podem estar confundidos no log.

Os valores completos e reproduzíveis desta tabela estão versionados em [`reports/experiment_summary.csv`](reports/experiment_summary.csv). O arquivo é regenerado por `python -m src.train` a partir dos mesmos resultados registrados no MLflow.

### Validação cruzada

O treinamento usa validação cruzada com `KFold` de 5 partes, embaralhamento reproduzível e seed 42. Em cada rodada, uma política nova é treinada por replay em 4 partes e avaliada, sem atualizar seu estado, na parte restante. Ao final, são calculadas as médias de conversão, taxa de casamento, eventos casados e conversões, além dos desvios-padrão das taxas de conversão e casamento.

Essa técnica verifica se as políticas mantêm desempenho em diferentes partes dos dados e é especialmente útil quando há poucos dados ou quando se quer avaliar a capacidade de generalização. O melhor braço histórico é calculado somente com os 4 folds de treino de cada rodada, evitando vazamento da validação. Após os 5 folds, uma nova política é treinada com todos os dados e salva em `models/` para uso pela API.

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
│   ├── feature_store.py         # Feature view local + contrato de metadados
│   ├── drift.py                 # Data drift (PSI e z-score) sobre a feature view
│   ├── hyperparam_search.py     # Grade de hiperparâmetros com registro no MLflow
│   ├── governance.py            # Contratos de segurança, viés e interpretabilidade
│   ├── pipeline.py              # Esteira local: dados → drift → busca → treino
│   ├── bandits.py               # Baseline, Epsilon-Greedy, Thompson Sampling + replay evaluation
│   ├── train.py                 # Comparação de políticas com tracking no MLflow
│   └── api.py                   # Etapa 5: serviço FastAPI de recomendação
├── models/                      # Estados (posteriores) das políticas treinadas
├── reports/
│   └── experiment_summary.csv  # Evidência reproduzível das métricas do MLflow
├── data/                        # raw/, processed/ e feature_store/ (gerados pelo pipeline)
├── .github/workflows/ci.yml     # CI/CD de código, dados e treino
├── requirements.txt
├── tests/                       # Testes de dados, políticas, NFR, drift e API
└── README.md
```

## Como executar localmente

Pré-requisito: Python 3.11+.

### Demonstração imediata da API

O artefato `models/thompson_sampling_contextual.json` usado pela API está versionado. Assim, após instalar as dependências, a recomendação pode ser demonstrada sem refazer o treinamento:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/mac: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn src.api:app --reload
```

Acesse `http://127.0.0.1:8000/docs` ou consulte `GET /health`, que deve retornar `{"status":"ok","policy_loaded":true}`.

### Reprodução completa do pipeline

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

# 7. Esteira completa (dados + Feature Store + drift + hiperparâmetros + treino)
python -m src.pipeline --raw-path tests/fixtures/bank_sample.csv --n-splits 3
```

Exemplo de chamada à API:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"age": 67, "job": "retired", "housing": "no", "loan": "no"}'
```

Resposta: braço recomendado, segmento, conversão esperada e posteriores Beta por braço (transparência da decisão).

O notebook das Etapas 1–4 pode ser aberto direto no Jupyter/VS Code: [notebooks/01_eda_e_bandits.ipynb](notebooks/01_eda_e_bandits.ipynb) (já executado, com outputs salvos).

### Deploy no Render

**API em produção**: <https://datathon-api-nyia.onrender.com> (interface web na raiz, Swagger em [/docs](https://datathon-api-nyia.onrender.com/docs)).

O repositório inclui um Blueprint [`render.yaml`](render.yaml) e uma imagem de produção enxuta. No painel do Render, escolha **New > Blueprint**, conecte este repositório e confirme a criação do serviço `datathon-api`. Não são necessárias variáveis de ambiente: o Render fornece `PORT` automaticamente e o artefato da política está versionado em `models/thompson_sampling_contextual.json`.

Após o deploy, valide `GET /health` e abra `/docs`. O health check retorna HTTP 503 quando a política não pode ser carregada, impedindo que uma instância degradada receba tráfego.

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

- **Parâmetros**: tipo de política, braços, `epsilon`, priors (`alpha`, `beta`), seed, dataset, nº de eventos e 5 folds de validação cruzada.
- **Métricas**: resultados de cada fold, médias de taxa de conversão, taxa de casamento, eventos casados e conversões, e desvios-padrão das taxas.
- **Artefatos**: estado JSON das posteriores de cada política (usado pela API - o mesmo artefato treinado é o que serve).

Se [`reports/best_hyperparams.json`](reports/best_hyperparams.json) existir, `python -m src.train` reutiliza os hiperparâmetros escolhidos pela busca automática. Além do backend local, o treinamento exporta [`reports/experiment_summary.csv`](reports/experiment_summary.csv). O estado contextual servido pela API fica em [`models/thompson_sampling_contextual.json`](models/thompson_sampling_contextual.json).

## Feature Store e data drift

`python -m src.data_prep` publica a feature view `campaign_context` em `data/feature_store/`: tabela corrente, schema, chaves de entidade (`segment`, `arm`) e estatísticas de referência. O contrato fica em `metadata.json`.

`python -m src.drift` compara um lote corrente com esse baseline:

- categóricas: Population Stability Index (alerta se PSI > 0,2)
- numéricas: z-score da média (alerta se |z| > 3)

O relatório é gravado em `reports/drift_report.json`. A esteira `python -m src.pipeline` publica a view, roda o drift e só então dispara busca + treino. Em CI, o job usa a fixture versionada [`tests/fixtures/bank_sample.csv`](tests/fixtures/bank_sample.csv) para não depender da UCI.

## Seleção automática de hiperparâmetros

`python -m src.hyperparam_search` percorre uma grade pequena e reproduzível:

- Epsilon-Greedy: `epsilon ∈ {0,05, 0,1, 0,2}`
- Thompson Sampling (global e contextual): priors `Beta(1,1)`, `Beta(2,2)` e `Beta(1,2)`

Cada candidato é avaliado por validação cruzada offline, registrado no MLflow (`datathon-bandit-hyperparams`) e o vencedor por política vai para `reports/best_hyperparams.json`.

## CI/CD

O workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) tem dois jobs:

1. **Código e NFR** - `pytest` de unidade, integração, segurança, viés e interpretabilidade.
2. **Dados e treino** - esteira com a fixture local, busca de hiperparâmetros e upload de `drift_report.json`, `best_hyperparams.json` e `experiment_summary.csv`.

## Testes não funcionais

- **Segurança**: o contrato da API recusa campos extras/sensíveis (`extra=forbid`); a Feature Store só materializa o contexto minimizado.
- **Viés**: `conversion_gap_by_segment` audita a disparidade de conversão entre segmentos.
- **Interpretabilidade**: cada `/recommend` devolve `explanation` com posteriores Beta e o motivo da escolha no segmento.

## Observabilidade e monitoramento

A API agora inclui três camadas de observabilidade para apoiar diagnóstico rápido de falhas, latência e degradação de negócio.

### 1) Logs estruturados

A aplicação registra eventos de requisição e de decisão de recomendação com `request_id`, método, rota, status, latência e contexto do segmento e do braço recomendado. Isso permite correlacionar log, alerta e execução de negócio em uma única cadeia.

Exemplos de eventos emitidos:

- `http_request_completed`
- `recommendation_generated`
- `health_check_ok`
- `health_check_degraded`

A configuração do logger fica em [src/api.py](src/api.py), com saída em stdout na forma:

```text
2026-08-24 12:00:00,000 INFO datathon.api request_id=7d3f... http_request_completed method=POST path=/recommend status_code=200 latency_ms=12.4
```

### 2) Tracing com OpenTelemetry

O projeto inicializa um `TracerProvider` e um span por requisição HTTP, além de um span específico para a geração da recomendação. Isso permite verificar:

- rota e status HTTP
- tempo de resposta
- request_id
- segmentação e braço recomendado
- variáveis de posterior Beta na etapa de decisão

A configuração usa exporter em console por simplicidade local, e pode ser ampliada para OTLP em produção. Em ambientes reais, basta apontar o exporter para um collector/Jaeger ou OTEL Collector.

### 3) Alertas mais ricos e dashboards

Os alertas em [prometheus/alerts.yml](prometheus/alerts.yml) foram expandidos para incluir:

- indisponibilidade da API
- taxa elevada de erro 5xx
- latência de requisição alta
- latência de recomendação alta
- política carregada como indisponível
- ausência de recomendações processadas
- colapso de conversão esperada

A dashboard de referência para Grafana está em [`monitoring/grafana/datathon_overview.json`](monitoring/grafana/datathon_overview.json). Ela expõe, em um painel único:

- taxa de requisições por segundo
- p95 de latência HTTP
- taxa de erro 5xx
- throughput de recomendações
- estado da política contextual

Arquivo de config do Prometheus: [`prometheus/prometheus.yml`](prometheus/prometheus.yml)
Arquivo de alertas: [`prometheus/alerts.yml`](prometheus/alerts.yml)

```bash
# 1) subir a API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 2) rodar o Prometheus localmente
docker run -d --name prometheus \
  -p 9090:9090 \
  -v "${PWD}/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml" \
  -v "${PWD}/prometheus/alerts.yml:/etc/prometheus/alerts/alerts.yml" \
  prom/prometheus
```

Acesse: `http://localhost:9090` para validar targets e alertas, e importei a dashboard em Grafana para acompanhar panos de disponibilidade, latência e negócio.

## Cobertura dos entregáveis (Etapas 0-7)

| Etapa | Evidência |
|---|---|
| 0 - Organização | `README.md`, `requirements.txt` e código modular em `src/` |
| 1 - Kaggle e EDA | Link da base e notebook `notebooks/01_eda_e_bandits.ipynb` executado |
| 2 - Preparação | `src/data_prep.py` e manifesto SHA-256 em `data/processed/data_manifest.json` |
| 3 - Baseline e adaptativos | Thompson Sampling, Epsilon-Greedy e baseline em `src/bandits.py` |
| 4 - Avaliação e Golden Set | Métricas versionadas e cinco casos documentados acima |
| 5 - Serviço demonstrável | FastAPI em `src/api.py` e artefato contextual versionado |
| 6 - Arquitetura em nuvem | Arquitetura AWS descrita na seção anterior |
| 7 - MLOps | Tracking MLflow em `src/train.py`, resumo em `reports/` e Prometheus em `prometheus/` |

## Governança e uso responsável de dados
Feature Store, drift, busca de hiperparâmetros, CI e Prometheus
- **Base legal/finalidade**: dados públicos e anonimizados (UCI/Kaggle) usados exclusivamente para fins educacionais; nenhum dado real de cliente, identificador, renda, gênero ou raça é utilizado.
- **Minimização**: o contexto usa apenas faixa etária e posse de produtos de crédito; colunas não usadas não saem do pipeline local.
- **Retenção**: dados brutos e CSV processado ficam locais (ignorados no Git) e podem ser regenerados; somente o manifesto de versão é mantido no repositório.
- **Humano no loop**: toda resposta da API sinaliza que decisões sensíveis exigem revisão humana antes da execução.
- **Limitações**: ver seção da base de dados e conclusões do notebook.
