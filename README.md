# 🚀 Modelo de concessão de crédito - Dados da Lending Club

Projeto de conclusão de curso da pós-graduação em Data Mining, análise de dados e inteligência artificial 



![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/lendingclub.png) 
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/credit.png) 

# Introdução:

## 📌 Objetivo
  O objetivo do trabalho é predizer a inadimplência na concessão de crédito feita por um clube de crédito dos Estados Unidos, a Lending Club. Uma plataforma que permite aos usuários emprestarem dinheiro a outros usuários.\
  A modelagem será realizada utilizando dados históricos de empréstimos passados, modelos estatísticos e algoritmos de Inteligência Artificial, que selecionarão as características mais relevantes que explicam o evento de inadimplência dos clientes.\
  Desta forma, a empresa poderá traçar estratégias de relacionamento, desenvolver réguas de cobrança e ações preventivas para minimizar de forma proativa sua taxa de inadimplência nas concessões de crédito.

## 🖇️ Contextualização do problema
  Lending Club é o maior mercado online de empréstimos, facilitando empréstimos pessoais, empréstimos comerciais e financiamento de procedimentos médicos. Os usuários podem acessar facilmente empréstimos com taxas de juros mais baixas por meio de uma interface online rápida.\
  Assim como para a maioria das outras empresas de crédito, conceder empréstimos a candidatos arriscados é a maior fonte de perda financeira (em inglês credit loss). Como usuários que descumprem o contrato (default) causam maior prejuízo para os credores, é necessário fazer o acompanhamento da taxa de inadimplência para os empréstimos oferecidos, bem como do apetite de risco e de eventuais estratégias de contenção.\
  O time de analytics da Lending Club recebeu a tarefa de desenvolver um modelo estatístico que preveja empréstimos que entrarão em default no futuro, de modo a minimizar a inadimplência dado diversos níveis de apetite de risco.\
  Esse modelo auxiliará o diretor e a instituição financeira a tomar decisões baseadas em dados, sem viéses socioeconômicos e culturais, bem como acelerar e automatizar o processo de tomada de decisão.
 
## 📋 Base original 
A base utilizada neste estudo foi obitida através de scrapping do site official da Lending Club, respeitando as políticas de concessão de dados do site.
A base fornece um conjunto muito grande e aberto de dados sobre as pessoas que receberam empréstimos por meio de sua plataforma, contemplando os anos 2007 a 2018 (para este estudo foram utilizados dados das safras de 2016 - 2018).\
Originalmente, este dataset contém 2.260.701 amostras e 151 variáveis com informações de diversos empréstimos já feitos no passado, bem como a variável status_emp (default), informando se o empréstimo foi ou não honrado.

## 🛠️ Filtros
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/filters.png)   

# 🔩 EDA - Análise Exploratória de Dados

### 1) Persona
- Profissão: maioria cargos de gerencia/diretoria e área técnica;
- A maior parte dos clientes está a mais de 10 anos no emprego atual;
- Prediomínio de usuários dos estados da Califórnia, Texas e Nova Yorque;
- Renda anual de 50mil a 90mil USD;
- Quanto à situação da casa, a maioria tem a casa hipotecada ou alugada.

![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/persona.png)   
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/emprego.png)   
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/anos_emprego.png)   
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/renda_anual.png)   
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/estado.png)   
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/score_fico.png)   

### 2) Variáveis transacionais
- As informações de saldo, ou transações, estão concentradas como variáveis transacionais.
- As distribuições para as variáveis saldo_cred, razão_saldo_lim_cred e taxa_uso_linha_rot seguem o comportamento da distribuição
normal, com assimetria a direita, dada a presença de outliers após o limite superior (como podemos ver nos boxplots).

![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/transact.png)
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/mesescont.png)  
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/linhacred.png)  

### 3) Variáveis do empréstimo
- A maior parte dos empréstimos acontece para consolidação de débito;
- Quanto ao prazo mais comum, a maioria opta por 36 meses (menor prazo), sendo esta categoria a com maior quantidade de default; 
- A faixa de valor predominante é de 5.000 – 10.000 usd

![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/prazo.png)
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/vlr_prest.png)  
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/tx_juros.png) 
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/score.png) 


### 4) Variável resposta
Status_emp:
- 1 = não pagou o empréstimo (Default)
- 0 = pagou o empréstimo
- A taxa de inadimplência na base de dados é de 22%.

![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/target.png) 


# ⚙️ Modelagem
## Tratamento das base de dados para modelagem

#### Divisão da base em treino e teste:
- Treino: anos de 2016 a 2017 (805756 linhas)
- Teste: 2018 (56318 linhas)
#### Tratamento de nulos:
- MeanMedianImputer: substituição de nulos pela média em variáveis numéricas
- CategoricalImputer: substituição de nulos por ‘missing’ em variáveis categóricas
- Numeric scaler: escalonamento de variáveis numéricas
- OneHotEncoder: criação de variáveis dummy para categóricas

## Modelos:
## 1 - Regressão Logística

O modelo seleciona as variáveis mais relevantes e estima um peso para cada uma de suas categorias, atribuindo para cada cliente a probabilidade de se tornar inadimplente.

  O modelo de regressão logística utilizado foi o Statsmodel, por permitir que se visualize as variáveis e o respectivo p-valor;\
  As variáveis foram selecionadas seguindo a abordagem clássica, removendo uma a uma e rodando o modelo novamente até que restassem apenas aquelas com p-valor < 0.05;\
  A regressão logística serviu como baseline para comparação com outros modelos.

![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/lr.png)
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/lr_metrics.png) 

## 2 - Árvore de Decisão
Classifica as observações pela combinação de características, por meio de uma árvore de classificação, que explique o evento de inadimplência.

![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/tree.png) 
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/tree_interpretation.png) 

## 3 - Random Forest
Ensemble – múltiplos algorítimos utilizados para obter melhores performances preditivas. Permite que exista uma estrutura muito mais flexível da resposta entre essas alternativas.\
Modelo tipo Bagging - Cria vários subsets de amostras de treino com reposição, e a saída do modelo é baseada em voto majoritário.

![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/rf_metrics.png) 

## 4 - Catboost
Ensemble – múltiplos algorítimos utilizados para obter melhores performances preditivas. Permite que exista uma estrutura muito mais flexível da resposta entre essas alternativas.\
Modelo tipo Boosting - Combina modelos fracos em fortes criando modelos sequenciais de tal forma que o modelo final tenha a maior acurácia .

![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/catb_metrics.png) 


# Desempenho dos modelos

O modelo com maior acurácia foi o CatBoost. Porém essa não é a única métrica que deve ser avaliada;\
Nenhum dos modelos tem valores altos de precision e recall juntos, então é necessário priorizar se teremos maiores taxas de falso negativo ou falso positivo;\
Como para este caso não queremos correr risco do cliente ser default (e gerar um prejuízo à empresa), optou-se por ter menores taxas de falsos negativos (recall);\
O modelo de eleição para o problema é o Random Forest.
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/plot_metrics.png) 
![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/models_metrics.png) 


# Otimização de hiperparâmetros

Foi feito o Grid Search para otimizar a métrica de recall da Random Forest\
Os hiperparâmetros escolhidos para otimização foram profundidade da árvore, mínimo de amostras por folha, número de estimadores (árvores), mínimo de amostras para divisão do nó.\
É uma forma de automatizar o processo de ajuste dos parâmetros de um algoritmo.

![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/hyper.png) 

# Shap Values
Útil para entendermos as decisões do modelo;\
Fornece a contribuição individual de cada uma das variáveis para o resultado do modelo

![img](https://github.com/emillypitman/tcc_fia/blob/main/imgs/shap.png) 

# Conclusão

O modelo que mais se ajustou à necessidade da equipe de negócios foi o Random Forest;\
Foi feito GridSearch para otimizar recall e garantir mais assertividade na identificação de clientes inadimplentes;\
Através do shap values foi possível destacar as variáveis que compõem a solução do modelo e debater com a equipe de negócios se estava fazendo sentido com o contexto.


