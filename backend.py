import re
import requests
from bs4 import BeautifulSoup
import streamlit as st

from collections import namedtuple

from json import JSONDecodeError
import json

import openai
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.document_loaders.youtube import YoutubeLoader

from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.summarize import load_summarize_chain,MapReduceDocumentsChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks import get_openai_callback

from langchain.chains import create_extraction_chain

from langchain.schema.output_parser import StrOutputParser

#Precio por 1 token
PRICING_MODEL_MAP = {
    "gpt-4" : {
        "input" : 0.03e-3,
        "output" : 0.06e-3,
    },
    "gpt-4-32k" : {
        "input" : 0.06e-3,
        "output" : 0.12e-3,
    },
    "gpt-3.5-turbo" : {
        "input" : 0.0015e-3,
        "output" : 0.002e-3,
    },
    "gpt-3.5-turbo-16k" : {
        "input" : 0.003e-3,
        "output" : 0.004e-3,
    },

}
TEMPLATE_RESUMEN = """ 
    Eres un asistente muy útil y experto en resumir una transcripción de un video.
    Tus objetivos son los siguientes:
        - Resumir en un máximo de dos párrafos el contenido de unos documentos dados por el usuario.

    Intenta siempre usar la mínima cantidad de texto posible.
    Usa la misma terminología que se usa en la transcripción.
    Responde únicamente con el resumen, no digas nada más.
    Responde en castellano.
    Vamos a ir paso a paso.
    """
TEMPLATE_TITULO = """ 
    Eres un asistente muy útil especializado en la creación de títulos creativos a partir de una transcripción de un video de Youtube.
    Tu objetivo es crear un único título que englobe los temas o conceptos más relevantes presentes en la transcripción proporcionada por el usuario.
    Crea el título en castellano.
    Crea un único título.
    El título debe ser corto y conciso.
    Responde sólo con el título.
    Piensa paso a paso.
    """
TEMPLATE_TEMAS = """ 
    Vamos a ir paso a paso.
    Eres un asistente muy útil que ayuda a sacar temas relevantes y sus marcas de tiempo a partir de una transcripción con marcas de tiempo de un video.
    Se te va a pasar una transcripcion de un video y un número de temas distintos a extraer de dicha transcripción.
    Tu objetivo es el siguiente:
        - Extraer los temas más importantes que se tratan en la transcripción y describirlos en una frase.
        - Extraer las marcas de tiempo en las cuales se empieza a tratar dichos temas. Extrae solo 1 marca de tiempo por tema.

    Da una breve descripción de cada tema después del nombre del tema.

    Ejemplos:
        - Tema: La historia del estoicismo
        - Descripción: Recorrido sobre la historia del estoicismo a lo largo de los años desde sus inicios hasta hoy en día.
        - Tiempo: 03:21

        - Tema: Beneficios del ayuno
        - Descripción: Ayunar aumenta los cuerpos cetónicos y la atención.
        - Tiempo: 11:45

    Usa la misma terminología que se usa en la transcripción.
    Responde únicamente con viñetas.
    Los temas tienen que ser importantes en la transcripción.
    Responde en castellano.
    Recuerda no repetir temas.
    Recuerda extraer solo 1 marca de tiempo por tema.

    %TRANSCRIPCION%
    {transcripcion}
    """
TEMPLATE_COMENT=""" 
    Eres un experto comentarista en Youtube.
    Escribes comentarios de videos de Youtube sobre muchos temas variados.
    Tus opiniones son respetuosas e interesantes.
    Eres creativo e invitas a la reflexión.

    Se te va a suministrar:
    - Unos temas y sus descripciones extraidos de un video de youtube.
    - Un resumen del video. 
    
    Tu misión es redactar un comentario u opinión interesante que incite a la reflexión para un video de youtube respondiendo a las preguntas:
    ¿ Qué opinas de los temas que se tratan ?
    ¿ Añadirías alguna información relevante sobre alguno de los temas ?

    Responde únicamente con el comentario, nada más.
    Utiliza un estilo informal típico de los videos de Youtube.
    Escribe en el mismo idioma que los temas y el resumen.
    Responde en 2 párrafos máximo.
    Ve al grano.

    %RESUMEN%
    {resumen}
    
    %TEMAS%
    {temas}   

    %TU OPINIÓN%
    """

TEMPLATE_COMMENT_PREGUNTAS =""" 
    Eres un experto comentarista en Youtube.
    Escribes comentarios de videos de Youtube sobre muchos temas variados haciendo preguntas sobre el contenido con el fin de obtener más información.
    Tus preguntas son respetuosas e interesantes.
    Eres creativo e invitas a la reflexión.

    Se te va a suministrar:
    - Unos temas y sus descripciones extraidos de un video de youtube.
    - Un resumen del video.
    - El autor del video   
    Tu misión es realizar algunas preguntas al autor del video sobre su contenido en forma de comentario.

    Responde únicamente con el comentario, nada más.
    Utiliza un estilo informal típico de los videos de Youtube.
    Escribe en el mismo idioma que los temas y el resumen.
    Sé conciso.

    %RESUMEN%
    {resumen}
    
    %TEMAS%
    {temas}

    %AUTOR%
    {autor}
    """

def _extract_video_id(url:str)->str:
    """Dado una url a un video de youtube, extrae el id del video.
    Esto es necesario para pasarselo al transcriptor de youtube

    Parameters
    ----------
    url : str
        _description_

    Returns
    -------
    str
        el video id
    """
    # Esto es una expresión regular que extrae el ID del video de YouTube
    youtube_id_match = re.search(r'(?<=v=)[^&#]+', url)
    youtube_id_match = youtube_id_match or re.search(r'(?<=be/)[^&#]+', url)
    video_id = (youtube_id_match.group(0) if youtube_id_match
                else None)
    return video_id

def _get_video_transcripts_with_timestamps(url:str)->str:
    """Dada una url a un video en youtube devuelve la transcripción
    con las marcas de tiempo correspondientes

    Parameters
    ----------
    url : str
        _description_

    Returns
    -------
    str
        transcripción con marcas de tiempo
    """
    #Parseamos el id del video
    video_id = _extract_video_id(url)
    if video_id is None:
        return ""
    #Sacamos la transcripcion en forma de dict con marcas de tiempo
    try:
        transcripcion = YouTubeTranscriptApi.get_transcript(video_id,languages=["es","en","fr"])
    except:
        return ""
    list_text_timestamps = []
    for piece in transcripcion:
        #Desglosamos hora minuto segundo
        hora = int(piece["start"] // 3600)
        minuto = int(piece["start"] // 60)
        segundo = int(piece["start"] % 60)
        #texto
        texto = piece["text"]
        #Lo metemos en la lista formateado
        list_text_timestamps.append(f"{minuto:02d}:{segundo:02d}- {texto}")

    text_with_timestamps = "\n".join(list_text_timestamps)
    return text_with_timestamps

@st.cache_data(show_spinner=False)
def _split_chunks(transcription:str)->list[Document]:
    """ Dada una transcripcion, devuelve una lista de varios documentos mas pequeños """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    docs = text_splitter.create_documents([transcription])
    return docs

def _get_split_tokens(docs:list[Document],llm:ChatOpenAI)->tuple[list[int],int]:
    """Dados unos documentos spliteados, devuelve una tupla con los
    tokens de cada documento en una lista y los tokens totales

    Parameters
    ----------
    docs : list[Document]
        lista de documentos spliteados de la transcripcion
    llm : ChapOpenAI
        el modelo de OpenAI instanciado

    Returns
    -------
    tuple[list[int],int]
        tupla con los tokens de cada documento en una lista y los tokens totales
    """
    tokens = []
    for doc in docs:
        tokens.append(llm.get_num_tokens(doc.page_content))
    return tokens,sum(tokens)

def get_autor_video(url:str)->str:
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True,language=["es","en","fr"])
    return loader._get_video_info()["author"]

@st.cache_resource(show_spinner=False)
def validar_yt_video_url(url:str,token_limit:int,_llm:ChatOpenAI)-> namedtuple:
    """
    Comprueba un enlace de un video a Youtube
    Comprueba que el video tenga transcripcion válida
    comprueba que no se supere el maximo numero de tokens establecido

    Args:
    url (str): The URL to check
    token_limit
    _llm
        
    Returns:
    namedtuple: Devuelve namedtuple con los campos: is_valid:bool, error_msg:str, docs y transcripcion
    """
    #Instanciamos la Namedtuple
    Respuesta = namedtuple("Respuesta", ["is_valid","error_msg","docs","transcripcion"])

    regex = r'^(?:http|ftp)s?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|' \
            r'[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    result = re.match(regex, url)
    if result:
        try:
            html = requests.get(url).text
            soup = BeautifulSoup(html, 'lxml')
            matches = soup.findAll("iframe", {"src": re.compile(r'(https\:\/\/youtu\.be\/|https?\:\/\/)?(.*?)($|\?)' )})
            if len(matches)>0:
                transcripcion = _get_video_transcripts_with_timestamps(url)
                if  transcripcion == "":
                    return Respuesta(False,"El video no tiene una transcripción válida o no se puede extraer su id.","","") #False,"El video no tiene una transcripción válida o no se puede extraer su id.","",""
                docs = _split_chunks(transcripcion)
                _, total_tokens = _get_split_tokens(docs,_llm)
                st.session_state["total_tokens"] = st.session_state.get("total_tokens",0) + total_tokens
                st.session_state["total_docs"] = len(docs)
                if total_tokens > token_limit:
                    return Respuesta(False,f"El número de tokens de la transcripción ({total_tokens}) excede el límite establecido de {token_limit}","","") #False,f"El número de tokens de la transcripción ({total_tokens}) excede el límite establecido de {token_limit}","",""
                st.session_state["url_valido"] = True
                st.session_state["docs"] = docs
                st.session_state["transcripcion"] = transcripcion
                return Respuesta(True,"", docs, transcripcion) #True,"", docs, transcripcion
            else:
                return Respuesta(False,"La url no parece apuntar a ningún video de Youtube","","") #False,"La url no parece apuntar a ningún video de Youtube","",""
        except Exception as exc:
            return Respuesta(False,f"La url no es válida.","","") #False,f"La url no es válida.","",""
    else:
        return Respuesta(False,"La url no es válida.","","") #False,"La url no es válida.","",""

def _create_chain_templates(*,llm:ChatOpenAI,template1:str,template2:str=None)->MapReduceDocumentsChain:
    system_message_prompt_map1 = SystemMessagePromptTemplate.from_template(template1)    
    human_template="Transcripción: {text}"
    human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map1, human_message_prompt_map])

    if template2:
        system_message_prompt_map2 = SystemMessagePromptTemplate.from_template(template2)
        chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map2, human_message_prompt_map])
        return  load_summarize_chain(llm,
                                chain_type="map_reduce",
                                map_prompt=chat_prompt_map,
                                combine_prompt=chat_prompt_combine,
                                #verbose=True,
                                )
    return load_summarize_chain(llm,
                                chain_type="map_reduce",
                                map_prompt=chat_prompt_map,
                                verbose=False,
                                )

@st.cache_data(show_spinner=False)
def _get_structured_data(text:str,schema:dict,_llm:ChatOpenAI)->list[dict]:
    
    structured_chain = create_extraction_chain(schema,_llm)
    try: 
        format_data = structured_chain.run(text)
    except JSONDecodeError as jerr:
        st.error("Lo sentimos, se ha producido un error. Vuelve a intentarlo más adelante.")
        st.stop()
    return format_data

@st.cache_data(show_spinner=False)
def validar_api_key(api_key:str)->tuple[bool,str,bool]:
    """Valida la api key y devuelve un bool en función de si hay acceso
    al modelo gpt-4 o no

    Parameters
    ----------
    api_key : str
        _description_

    Returns
    -------
    tuple[bool,str,bool]
        [si la key es válida,el mensaje de error en caso de haber, si hay acceso a gpt-4]
    """
    url = "https://api.openai.com/v1/models"  # Endpoint de prueba que devuelve detalles de los modelos a los que tienes acceso

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }           
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("La API key es válida.")
            #Comprobamos si acceso a gpt-4
            json_response = json.loads(response.text)
            if "gpt-4" in [diccionario["id"] for diccionario in json_response["data"]]:
                #Hay acceso a gpt-4
                return True,"",True
            else:
                #No tenemos acceso a gpt-4
                return True,"",False
        else:
            print("La API key no es válida.")
            return False,"La API key no es válida.",False
    except requests.exceptions.RequestException as e:
        print("Ocurrió un error al validar la API key:", e)
        return False, f"Ocurrió un error al validar la API key: {e}",False
    
@st.cache_data(show_spinner=False)
def get_summary(template:str,_docs:str,_llm:ChatOpenAI)->str: 

#with get_openai_callback() as cb:
    try:
        chain_resumen:MapReduceDocumentsChain = _create_chain_templates(llm=_llm,template1=template,template2=template)
    except openai.RateLimitError as e:
        st.error(f"Has excedido tu cuota de openai para esta api key. Revisa tu plan: {e}")
        st.stop()

    resumen:str = chain_resumen.run({"input_documents": _docs})
    #st.session_state["total_cost"] = st.session_state.get("total_cost",0) + cb.total_cost
    #st.session_state["total_tokens"] = st.session_state.get("total_tokens",0) + cb.total_tokens
    return resumen

@st.cache_data(show_spinner=False)
def get_title(template:str,_docs:str,_llm:ChatOpenAI)->str:    

    with get_openai_callback() as cb:
        try:
            chain_resumen:MapReduceDocumentsChain = _create_chain_templates(llm=_llm,template1=template,template2=template)
        except openai.RateLimitError as e:
            st.error(f"Has excedido tu cuota de openai para esta api key. Revisa tu plan: {e}")
            st.stop()

        resumen:str = chain_resumen.run({"input_documents": _docs})
        st.session_state["total_cost"] = st.session_state.get("total_cost",0) + cb.total_cost
        st.session_state["total_tokens"] = st.session_state.get("total_tokens",0) + cb.total_tokens        
    return resumen

def _get_lista_temas_y_marcas(temas:str)->tuple[list,list,list]:
    """Coge la lista de temas y marcas en str y desglosa los temas, las marcas y las descripciones

    Parameters
    ----------
    temas : str
        _description_

    Returns
    -------
    tuple[list,list,list]
        Devuelve lista de marcas, temas, descripciones
    """
    #print(temas)
    titulos_temas = []
    [titulos_temas.extend(tema.split("\n")) for tema in temas.split("\n\n")]
    lista_de_temas = [tema[8:] for tema in titulos_temas[::3]]
    lista_de_marcas = [tema[12:] for tema in titulos_temas[2::3]]   
    lista_de_descripciones = [tema[17:] for tema in titulos_temas[1::3]]

    return lista_de_marcas, lista_de_temas, lista_de_descripciones

@st.cache_data(show_spinner=False)
def get_themes_and_stamps(template:str,_docs:list[Document],_llm:ChatOpenAI)->tuple[list,list,list]:
    """ Saca lista de marcas, lista de temas y lista de comentarios """    

    prompt_sacar_temas = ChatPromptTemplate.from_template(template)

    chain_sacar_temas = (
    prompt_sacar_temas
    | _llm
    | StrOutputParser()
)
    #with get_openai_callback() as cb:
    output = st.empty()
    output.write("Se están sacando los temas de la transcripción.\nEsta operación puede tardar unos minutos...")
    temas = chain_sacar_temas.invoke({"transcripcion" : _docs})
    #st.session_state["total_cost"] = st.session_state.get("total_cost",0) + cb.total_cost
    #st.session_state["total_tokens"] = st.session_state.get("total_tokens",0) + cb.total_tokens
    output.empty()

    marcas, temas, descripciones = _get_lista_temas_y_marcas(temas)    
    return marcas, temas, descripciones

@st.cache_data(show_spinner=False)
def _comment_from_ia(template:str,temas:str,resumen:str,_llm:ChatOpenAI,autor:str)->str:
    """ Función para obtener 1 comentario de la IA a partir de una lista de temas y descripciones de los temas. """      

    prompt_opinar =  ChatPromptTemplate.from_template(template)

    chain_sacar_temas = (
    prompt_opinar
    | _llm
    | StrOutputParser()
    )
    with get_openai_callback() as cb:
        output = st.empty()
        output.write("Se está generando un comentario de los temas.\nEsta operación puede tardar unos minutos...")
        comentario = chain_sacar_temas.invoke({"temas":temas,"resumen":resumen,"autor":autor})
        st.session_state["total_cost"] = st.session_state.get("total_cost",0) + cb.total_cost
        st.session_state["total_tokens"] = st.session_state.get("total_tokens",0) + cb.total_tokens
        output.empty()

    return comentario

def estimar_coste_generacion(
        templates:list[str],
        modelos:list[ChatOpenAI],
        hay_titulo:bool,
        hay_comentario:bool,
        transcripcion_tokens:int)-> tuple[int,float]:
    
    """función para estimar los costes de la generación
    """
    output_tokens_estimacion = {
        "titulo" : 100,
        "resumen" : transcripcion_tokens // 10,
        "temas" : transcripcion_tokens // 75,
        "comentario" : 850,
    }

    assert len(modelos) == len(templates), "Las cantidades de modelos y de plantillas debe coincidir !"
    # El orden será: titulo-resumen-temas-comentario
    tokens_estimados = [] #(input,output)
    coste_estimado = []

    # primero iteramos para sacar los tokens de cada template
    for (template, modelo,key) in zip(templates,modelos,output_tokens_estimacion.keys()):
        #calculamos tokens y metemos en lista en orden
        #inputs
        tokens_inputs = modelo.get_num_tokens(template) + transcripcion_tokens
        #outputs
        tokens_outputs = output_tokens_estimacion[key]
        tokens_estimados.append([tokens_inputs,tokens_outputs])
            
    # Si es el comentario, le pasamos a la template el resumen y los temas, no los docs, restamos al último indice
    tokens_estimados[-1][0] = tokens_estimados[-1][0] - transcripcion_tokens + output_tokens_estimacion["resumen"] + output_tokens_estimacion["temas"]

    # Calculamos costes en funcion de modelo e input/output
    for (modelo,tupla_tokens) in zip(modelos,tokens_estimados):
        #coste del resumen
        coste_estimado.append(
            tupla_tokens[0]*PRICING_MODEL_MAP[modelo.model_name]["input"]\
            + tupla_tokens[1]*PRICING_MODEL_MAP[modelo.model_name]["output"]) 

    # Comprobamos opciones seleccionadas
    if not hay_titulo:
        del tokens_estimados[0]
        del coste_estimado[0]
    if not hay_comentario:
        del tokens_estimados[-1]
        del coste_estimado[-1]
    
    return sum([sum(tup) for tup in tokens_estimados]), round(sum(coste_estimado),4)

