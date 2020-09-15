##########################################################################################################################
#  12/09/2020 - Trabalho de Conclusão de Curso - Tera Treinamentos Profissionais 
#               Sistema de Recomendação de Vinhos 
#
##########################################################################################################################


############################
# Importando Libraries
############################
import pandas as pd
import numpy as np

# Natural Languade Tool Kit
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

## TFIDF libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## Streamlit
import streamlit as st 
from PIL import Image

############################
# Carregando dataset
############################
dados = pd.read_csv('vinhos.csv' , sep = ';')

# Valor default de Similaridade
limite = 0.65   

############################
# Array numpy de Paises
############################
paises = dados.pais.unique()
#--- Problema: existe um pais denominado 'Espumante'
pos = np.where(paises == 'Espumante')
paises = np.delete(paises, pos)
#--- Fim do Problema
paises = np.sort(paises)

############################
# Array numpy de Tipos
############################
tipos = dados.tipo.unique()
tipos = np.sort(tipos)

############################
# Array numpy de Descricao
############################
descricao = dados.descricao.unique()
#--- Problema: existe uma descricao '750 ml'
pos = np.where(descricao == '750 ml')
descricao = np.delete(descricao, pos)
#--- Fim do Problema
descricao = np.sort(descricao)

############################
# Array numpy de Uva1
############################
uvas = dados.uva1.unique()
uvas = np.sort(uvas)

#####################################################################################################################
##  ^.^  F U N C O E S  ^.^                                                                                         #
#####################################################################################################################

#--------------------------------------------------------------------------------------------
### Função construida para contornar o problema de sort_values(by=['uva1','score']),
#--------------------------------------------------------------------------------------------
## Funcao que contorna o problema de sort_values(by=['uva1','score']),
def build_recom(df):

    grupo_uvas = df.groupby('uva1')

    build_df = pd.DataFrame()
    for name, grupo in grupo_uvas: # name = uvas1 / grupo = dataframe associado !

        select = (df.uva1 == name)
        df_temp = df[select].sort_values(by=['score'],ascending=False)
        build_df = pd.concat([build_df,df_temp], axis=0)

    return build_df

#--------------------------------------------------------------------------------------------
### Função construida para avaliar o array numpy, extraido da matrix de cosine_similarity ###
#--------------------------------------------------------------------------------------------
## Função construida para avaliar o array numpy, extraido da matrix de cosine_similarity
## Evitar o indice "itself"  
## Estabelecer uma nota de corte (limite)
## Estabelecer a quantidade de registros maximos a serem capturados (np.arange(10))
## Retorna uma lista de indices com melhor medida de similaridade.

def favalcosine(medida,idx):

    # variavel declarada como global, 12/9/20   limite = 0.65
    result = []
    
    for cnt in np.arange(10):
    
        ind = cnt - 10  ## ind vai de -10 até -1 !!  
        index = medida.argsort()[ind] # Retorna os INDICES que ordenam o array. 
        measure = medida[index]       # Como a ordem é ascendente, precisa começar do final do array. 

        if index == idx:
            continue

        if (measure > limite):
            result.append([index,measure])

    return result


###############################
#  Comandos STREAMLIT - Titulo
###############################
st.title('Sistema de Recomendação de Vinhos')

############################
# Adicionar Imagem
############################
img = Image.open('Chateaux.jpg')
st.image(img, caption='Bordeaux',use_column_width=True)

############################
# Inicio da Selecao
############################
st.subheader('Forneça as características do(s) vinho(s) desejado(s).')

############################
# Seleção de Pais
############################
st.write("""
    ### Por favor, selecione o Pais.
    """)
sel_pais = st.selectbox('Paises' , paises )

############################
# Seleção de Tipo
############################
st.write("""
    ### Por favor, selecione o Tipo.
    """)
sel_tipo = st.selectbox('Tipos' , tipos )

############################
# Seleção de Descricao
############################
st.write("""
    ### Por favor, selecione a Descrição.
    """)
sel_desc = st.selectbox('Descrição' , descricao )

############################
# Seleção de Varietais
############################
st.write("""
    ### Por favor, selecione os Varietais.
    """)
sel_uvas = st.selectbox( 'Varietais', uvas )    
# sel_uvas = st.multiselect( 'Por Favor, selecione os Varietais.', uvas )    
# st.write('sua selecao:' , sel_uvas )

############################
# Range de Preços - Slider
############################
preco_de = st.slider('Escolha o preco inicial',0,500,0,10)
preco_ate = st.slider('Escolha o preco final',preco_de,500,preco_de,10)

##########################################
# Parametro limite - grau de similaridade
##########################################
par_limite = st.slider('Grau de Similaridade',0.50,1.00,0.50)
# st.slider(label, min_value, max_value, value=none, step=none)
limite = par_limite
################################
# Parametro limite - atualizado
################################


##########################################
# Botão de Acionamento de todo o modelo
##########################################
if st.button('Recomendar'):

    #--- RESET da preferencia
    dados['consumo'] = 0

    #--------------------------
    # Construcao de filtro
    #--------------------------
    select = (dados.pais == sel_pais) & (dados.tipo == sel_tipo) & (dados.descricao == sel_desc) & (dados.uva1 == sel_uvas) & (dados.price >= preco_de)
    if preco_ate > 0:
        select = select & (dados.price <= preco_ate)
    df_filtro = dados[select]

    #=======================================
    # Define a posicao da coluna 'consumo'
    #=======================================
    my_idx = dados.columns
    pos_consumo = my_idx.get_loc('consumo')

    for ind, dfseries in df_filtro.iterrows():
        dados.iloc[ind,pos_consumo] = 1

    #---- Apresentando a seleção
    select = (dados.consumo == 1)
    colunas = ['nome','preco','pais','tipo','descricao','uvas','ano']
    df_temp = dados[select][colunas].sort_values(by=['uvas'], ascending=False)

    #--- Precisa verificar se existe algo a apresentar !!
    tam_temp = df_temp.shape[0]

    if tam_temp == 0:
        st.write('Lamentamos informar que sua seleção não retornou resultados. Tente novamente!')
        st.stop()
    else:
        st.write('Sua Seleção de Consumo é apresentada abaixo:')
        df_temp  #

    #############################################################################################################
    ## Função de Recomendação, baseada no conceito de Vector Space Model, onde um documento de 
    ## texto é transformado em uma representação vetorial, em um espaço multi dimensional.
    ## Após a transformação, passamos a empregar a Medida de Similaridade de Coseno, afim de determinar
    ## quais Observações mais se aproximam do conjunto de Preferencias do usuário, dentro do catalogo.
    #############################################################################################################

    # TfidfVectorizer não possui um conjunto interno de stop words em Portugues, mas aceita uma lista 
    # de palavras como parametro. Dessa forma, vamos gerar as stopwords em Portugues, usando o NLTK.

    # Armazena as stop words em Portugues em uma variavel
    stopwords = set(stopwords.words('portuguese'))
    # Lembrando que set() remove itens duplicados !

    #--- Inserindo uma coluna para score
    dados['score'] = float(0)

    #--- Separar um dataframe com a coluna de texto agregado
    meu_df = dados[['texto']]

    #----------------------------------------------------------------------------------
    #  ***  Acionamento do Modelo para transformação da coluna de texto [agregado]  ***
    #----------------------------------------------------------------------------------
    # Inicializando o modelo Vetorizado TF-IDF
    tf = TfidfVectorizer(analyzer='word',min_df=0,stop_words=stopwords)
    # Observe que é passada a lista de palavras para o parametro stop_words.  
    # Nativamente, o modelo Tfidf não possui stop words que não seja em Ingles.

    # Usa o modelo para transformar a coluna de texto
    tfidf_matrix = tf.fit_transform(meu_df['texto'])    
    
    #-------------------------------------------------------------------------------------------
    #  *** Acionando a Medida de Similaridade de Coseno, com base na matriz tfidf já calculada.
    #-------------------------------------------------------------------------------------------
    ## A estratégia aqui é : usar TODA a matriz modelada. A etapa seguinte deve identificar 
    ## quais são os registros que nos interessam, aquelas que já foram marcados com Consumo = 1,
    ## e fazer a iteração a partir desse sub conjunto.
    ## IMPORTANTE: o resultado da aplicação de cosine_similarity é também uma matriz, onde
    ## cada linha representa cada Observação do dataset, e as colunas fornecem a Medida de Similaridade
    ## entre todas as demais Observações do dataset. Óbvio que a Similaridade entre a Observação e ela mesma é igual a 1.
    ## Na aparencia, lembra um pouco a matriz de correlação.

    medida = cosine_similarity(tfidf_matrix,tfidf_matrix) 

    #-------------------------------------------------------------------------------------
    #  ***  Fazendo a iteração do conjunto de registros marcados como Consumo = 1   ***
    #-------------------------------------------------------------------------------------
    select = (dados.consumo == 1)
    meu_consumo = dados[select]

    #--- Inicio da Iteracao ---
    results = []
    scores  = []
    backtrack = []

    # Lembrando que meu_consumo contem apenas os registros do Consumo == 1

    # Somente para preenchimento do backtrack !!!
    for idx, row in meu_consumo.iterrows():
        #--- Armazena previamente TODOS os indices do Consumo !!!
        if idx not in backtrack:
            backtrack.append(idx)

    # Agora, sim, tudo pronto, salva guardas garantidas
    for idx, row in meu_consumo.iterrows():

        ## Para cada Observacao, envia a parte correspondente da matriz de similaridade
        result_parcial = favalcosine(medida[idx],idx) 
            
        #------------------------------------------------
        # Formato da Lista result_parcial, como retorno 
        # da função favalcosine: 
        #   parcial_index  = result_parcial[0]
        #   parcial_scores = result_parcial[1]
        #------------------------------------------------
                    
        for ind, scr in result_parcial:

            if ind not in results:
                if ind not in backtrack:
                    results.append(ind)
                    scores.append(scr)
    #--------------------------------------------
    #  ***  Fim da iteração do Consumo = 1   ***
    #--------------------------------------------
   
    #--- Loop que faz a montagem do dataframe final ---
    df_recom = pd.DataFrame()

    for ind in results:
                
        select = ( dados.index == ind)
        df_temp = dados[select]
            
        df_recom = pd.concat([df_recom, df_temp], axis=0)

    #--- Atribui a coluna [score]; lembrando que scores é uma lista ---
    df_recom.score = scores

    #--- FIM da Iteracao ---
    #-----------------------------------------------------
    #  ***  Construindo o formato do Resultado Final   ***
    #-----------------------------------------------------
    #===============================================
    # Aplicacao do Filtro Final - Somente Varietal #
    #===============================================
    filtro_variet = (df_recom.varietal == 'S')

    df_tmp = df_recom[filtro_variet].sort_values(by=['uva1'],ascending=False)

    # Funcao para reordenar o dataset, dentro de cada ['uva1']
    df_variet = build_recom(df_tmp)
                
    #============================================
    # Aplicacao do Filtro Final - Somente Blend #
    #============================================
    filtro_blend = (df_recom.varietal == 'N')

    df_tmp = df_recom[filtro_blend].sort_values(by=['uva1'],ascending=False)

    # Funcao para reordenar o dataset, dentro de cada ['uva1']
    df_blend = build_recom(df_tmp)

    #=================
    #  Concatenação  #
    #=================
    data_final = pd.concat([df_variet , df_blend], axis=0 ) 
    # Quero que as Uvas de VARIETAL PRINCIPAL fiquem no topo da recomendacao

    colunas = ['nome','preco','pais','tipo','descricao','uvas','ano','rating','score']
    df_resultado = data_final[colunas]
    #st.dataframe(my_selection)

    #--- Precisa verificar se existe algo a apresentar !!
    tam_resul = df_resultado.shape[0]

    if tam_resul == 0:
        st.write('Lamentamos informar que a Recomendação não retornou resultados. Tente novamente!')
        st.stop() 
    else:
        st.write('Sua Recomendação é apresentada abaixo:')
        df_resultado  #
