import streamlit as st
import backend as B
import time
from typing import Literal, Union
import json
from io import BytesIO
import youtube_model as ym
import llm_model as lm

from collections import namedtuple

## Configuración de la app ##
st.set_page_config(
    page_title="Youtube SyTiCo",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="expanded",
)

## Constantes ##
LIMIT_TOKEN = 10_000
HEADER_EMOJI = ""
SUBHEADER_EMOJI = ""
LISTA_MODELOS_STD = ("gpt-3.5-turbo","gpt-4")
LISTA_MODELOS_LARGE = ("gpt-3.5-turbo-16k","gpt-4")
INDICE_BASE = 0

## Funciones auxiliares ##
def init_outputs()->None:
    "Inicial algunas variables de sesión necesarias"
    st.session_state["outputs"] = {}
    st.session_state["activador_stream"] = st.session_state.get("activador_stream",False)
    st.session_state["url_valido"] = False
    st.session_state["transcripcion"] = ""
    st.session_state["docs"] = []
    st.session_state["total_cost"] = 0
    st.session_state["total_tokens"] = 0
    st.session_state["coste_estimado"] = 0
    st.session_state["tokens_estimados"] = 0

def borrar_cache():
    st.session_state["outputs"] = {}
    st.cache_data.clear()

def stream_output(
        texto:str,
        cadencia:float=0.02,
        encabezado:Union[None,Literal["header","subheader","title","write"]]="write"
        )->None:
    """Streamea la respuesta del LLM con una cadencia determinada

    Parameters
    ----------
    texto : str
        El texto a streamear
    cadencia : float, optional
        en segundos cuanto espera antes de mostrar la siguiente cadena de texto, by default 0.02
    """
    FUNC_MAPPING = {
        "header" : st.header,
        "title" : st.title,
        "subheader" : st.subheader,
        "write" : st.write,
    }

    frase = ""
    output = st.empty()
    for char in texto:
        frase += char
        with output:
            FUNC_MAPPING[encabezado](frase)
            time.sleep(cadencia)
    
def mostrar_outputs_titulo()->None: 
    """Muestra el titulo.    """
    titulo = st.session_state["outputs"].get("Titulo","")
    if st.session_state["activador_stream"]:
        stream_output(titulo,encabezado="title")
    else:
        st.title(titulo)

def mostrar_outputs_resumen()->None: 
    """Muestra el resumen.    """        
    resumen = st.session_state["outputs"].get("Resumen","")
    if resumen:
        st.header("Resumen")
    if st.session_state["activador_stream"]:
        stream_output(resumen)
    else:
        st.write(resumen)

def mostrar_outputs_temas(descripcion_temas:bool=False)->None:
    """Muestra los outputs"""    
    tdo_junto = st.session_state["outputs"].get("todo_junto",[])
    if tdo_junto:
        st.header("Temas")
    for marca,titulo,descripcion in tdo_junto:
        if st.session_state["activador_stream"]:
            stream_output(f"{marca} {titulo}",encabezado="subheader")
            if descripcion_temas:
                stream_output(descripcion,encabezado="write")
        else:
            st.subheader(f"{marca} {titulo}")
            if descripcion_temas:
                st.write(f":blue[{descripcion}]")

def juntar_temas_descripciones(temas:list,descripciones:list)->str:
    salida = "".join(
        [f"""tema {idx}: {tema}\ndescripción: {desc}\n\n""" 
        for idx,(tema,desc) 
        in enumerate(zip(temas,descripciones),start=1)])
    return salida

def mostrar_comentario()->None:
    comentario = st.session_state.get("outputs",{}).get("Comentario","")
    if comentario:    
        st.header("Comentario YouTube")
    if st.session_state["activador_stream"]:
        stream_output(comentario)
    else:
        st.write(comentario)

def crear_comentario_yt_temas(lista_todo_junto:list[tuple])->str:
    """Función para crear un string con el fin de enviarlo a comentario de youtube

    Parameters
    ----------
    lista_todo_junto : list
        la lista con las tuplas marcas, temas, descripcion

    Returns
    -------
    str
        todo en un string
    """
    comentario = ""
    for marca,titulo,descripcion in lista_todo_junto:
        comentario += f"""[{marca}] {titulo}\n{descripcion}\n\n"""

    return comentario

def cargar_json(json_file_bytes:BytesIO)->json:
    """Carga un archivo json en bytes y devuelve el archivo json

    Parameters
    ----------
    json_file_bytes : bytes
        _description_

    Returns
    -------
    json
        _description_
    """
    json_file = json_file_bytes.read()
    content_str = json_file.decode('utf-8')
    data = json.loads(content_str)
    return data

def credenciales_validas(json_file:json)->tuple[bool,str]:
    """
    Valida que el contenido en bytes de las credenciales contenga las claves necesarias.

    :param content_bytes: Contenido del archivo JSON de credenciales en formato bytes.
    :return: True si es válido, False en caso contrario. y str con contenido sobre el tipo de credenciales
    """
    required_keys = [
        "client_id",
        "project_id",
        "auth_uri",
        "token_uri",
        "auth_provider_x509_cert_url",
        "client_secret",
        "redirect_uris"
    ]
    info = ""
    try:
        if "installed" in json_file:
            credentials_data = json_file["installed"]
            info = "desde app de escritorio."
        elif "web" in json_file:
            credentials_data = json_file["web"]
            info = "desde aplicación web. Verifica los campos 'redirect_uris' y 'javascript_origins'. "
        else:
            st.error("El contenido no contiene la clave 'installed' o 'web'.")
            return False, info

        for key in required_keys:
            if key not in credentials_data:
                st.error(f"Falta la clave '{key}' en el contenido de credenciales.")
                return False, ""

        return True, info

    except json.JSONDecodeError:
        st.error("El contenido no es un JSON válido.")
        return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False


if __name__ == '__main__':
    # st.session_state
    st.session_state["activador_stream"] = False
    # iniciamos las variables de entorno
    st.session_state["valida_api_key"] = False

    if st.session_state.get("outputs") is None:
        init_outputs()

    with st.sidebar:
        st.header(f"{HEADER_EMOJI} ⚙️ Configuración del modelo OpenAI {HEADER_EMOJI}")
        api_key = st.text_input(
                    label="OpenAI Api Key",
                    type="password",
                    placeholder="Introduce tu api key",
                    help="Para crear una API key visita la web de OpenAI en: https://platform.openai.com/account/api-keys ")
        
        st.info("Si el acceso a gpt-4 no está disponible para tu Api Key, se usará\
                automáticamente gpt-3.5-turbo")
        
        # Validamos api_key y verificamos si acceso a gpt-4
        if api_key:
            key_valida, err_key,acceso_gpt4 = B.validar_api_key(api_key)
            if not key_valida:
                st.error(err_key)
            else:
                if acceso_gpt4:
                    st.success("Tienes acceso a gpt-4")
                else:
                    st.warning("No tienes acceso a gpt-4")
                st.session_state["valida_api_key"] = True

        # Opciones para elegir el modelo
        st.subheader(f"{SUBHEADER_EMOJI} Escoger modelos {SUBHEADER_EMOJI}")
        
        modelo_titulo = st.radio(
            "Modelo para el **título**",
            LISTA_MODELOS_STD,
            index=INDICE_BASE,
            help="Para ver los precios ir a https://openai.com/pricing\nEl modelo gpt-4 se usará siempre si está disponible."
        )
        modelo_resumen = st.radio(
            "Modelo para el **resumen**",
            LISTA_MODELOS_STD,
            index=INDICE_BASE,
            help="Para ver los precios ir a https://openai.com/pricing\nEl modelo gpt-4 se usará siempre si está disponible."
        )
        modelo_temas = st.radio(
            "Modelo para los **temas y sus descripciones**",
            LISTA_MODELOS_LARGE,
            index=INDICE_BASE,
            help="Para ver los precios ir a https://openai.com/pricing\nEl modelo gpt-4 se usará siempre si está disponible."
        )
        modelo_comentario = st.radio(
            "Modelo para el **comentario**",
            LISTA_MODELOS_STD,
            index=INDICE_BASE,
            help="Para ver los precios ir a https://openai.com/pricing\nEl modelo gpt-4 se usará siempre si está disponible."
        )
        if api_key and key_valida:                
                # Instanciamos los modelos
                gestor_modelos = lm.GestorModelos(
                    [modelo_titulo,
                    modelo_resumen,
                    modelo_temas,
                    modelo_comentario],
                    acceso_gpt4,
                    api_key,
                )
                llm_titulo, llm_resumen,\
                llm_temas, llm_comentario = gestor_modelos.instanciar_modelos()                       
                
        descripcion_temas = False
        st.divider()
        st.header(f"{HEADER_EMOJI} ⏳ Opciones de procesamiento {HEADER_EMOJI}")
        token_limit = st.number_input(
            label="Límite max tokens",
            value=LIMIT_TOKEN,
            help="""El límite máximo de tokens permitido para la transcripcion.
            Si la transcripción excede de este límite, la aplicación se detendrá y mostrará un mensaje de error.""",
            max_value=LIMIT_TOKEN+2500,
        )
        mostrar_titulo = st.checkbox(
        "Generar el título",
        value=False,
        )        
        comentarios_auto = st.checkbox(
        "Generar comentario",
        value=False,
        )
        
        st.subheader(f"{SUBHEADER_EMOJI} Comentar en Youtube {SUBHEADER_EMOJI}")
        # Plataforma de carga de credenciales de google youtbe
        st.info("Para poder comentar en youtube hace falta cargar un archivo de credenciales google")
        json_file_bytes = st.file_uploader(
            "Carga tu archivo de credenciales",
            help="""
                Para conseguir tu archivo de credenciales sigue los siguientes pasos:\n
                1. Ve a https://console.cloud.google.com/welcome?project=compact-window-380615\n\
                2. Crea un proyecto nuevo\n
                3. Habilita la YouTube Data API v3 para ese proyecto.\n
                4. Crea credenciales (OAuth 2.0 Client ID). Asegúrate de configurarlo para una aplicación de escritorio.\n
                5. Descarga el archivo de credenciales.""",
                type=["json"])
        # Validamos el archivo
        if json_file_bytes:
            st.session_state["credenciales_json"] = cargar_json(json_file_bytes)
            validar_credenciales, info = credenciales_validas(st.session_state["credenciales_json"])
            if validar_credenciales:
                st.success(f"Credenciales válidas {info}")
        
        st.divider()

        st.header(f"{HEADER_EMOJI} 🔎 Opciones de visualización {HEADER_EMOJI}")           
        streamear_respuesta = st.checkbox(
            "Streamear respuesta",
            value=False,
            help="Si está seleccionado el texto va apareciendo letra a letra."
        )
        if streamear_respuesta:
            st.session_state["activador_stream"] = True

        descripcion_temas = st.checkbox(
        "Mostrar la descripción de los temas",
        value=True
        )           
        
        #if st.session_state.get("outputs") != {}:
        #    borrar = st.button("Borrar",on_click=borrar_cache,help="Borra la generación")
        st.divider()
        
    st.title("🎥 :red[You]tube SyTiCo")
    st.header(":green[Sy]nthesize :green[Ti]mestamp & :green[Co]mment")
    col1, col2 = st.columns(2,gap="large")

    with col1:
        st.write("""
        Aplicación para sintetizar el contenido de un video de youtube. También está la posibilidad de \
                agregar comentarios generados por la IA a cada uno de los temas y comentar automáticamente\
                en youtube.\n
        Cómo funciona:
        1. Introduce tu api key de OpenAI
        2. Escoge los parámetros de configuración.
        3. Si quieres comentar en Youtube automáticamente, carga las credenciales json.
        4. Introduce una URL válida a un video de youtube y pulsa Enter.
        5. Procesar
        6. Si te gusta la generación, comenta en Youtube.
        """)

    with col2:
        url = st.text_input(
            label ="Introduce una URL de Youtube",
            on_change=init_outputs,
            )

        procesar = st.button("Procesar",on_click=borrar_cache)

    # Validamos la url
    if url and st.session_state["valida_api_key"]:
        if not st.session_state["url_valido"]:
            with st.spinner("Validando url..."):
                respuesta:namedtuple = B.validar_yt_video_url(url,token_limit,gestor_modelos._get_modelo()) #url_valido,url_err,_,_
            if not respuesta.is_valid:
                st.error(respuesta.error_msg)
                st.stop()
        if st.session_state["url_valido"]:
            # sacamos el autor
            st.session_state["autor"] = B.get_autor_video(url)
            # instanciamos el tracker de las publicaciones de comentario y temas
            st.session_state[url]={
                "publicar_temas" : False,
                "publicar_comentario" : False}
            if procesar:
                if mostrar_titulo:
                    with st.spinner("Sacando el título..."):
                        titulo = B.get_title(B.TEMPLATE_TITULO,st.session_state["docs"],llm_titulo)
                        st.session_state["outputs"]["Titulo"] = titulo
                        st.session_state["activador_stream"] = True                

                with st.spinner("Sacando el resumen..."):
                    resumen = B.get_summary(B.TEMPLATE_RESUMEN,st.session_state["docs"],llm_resumen)
                st.session_state["outputs"]["Resumen"] = resumen
                st.session_state["activador_stream"] = True
                
                with st.spinner("Sacando temas"):
                    marcas, temas, descripciones = B.get_themes_and_stamps(
                        B.TEMPLATE_TEMAS,
                        st.session_state["docs"],
                        llm_temas)                     

                if comentarios_auto:
                    temas_descripciones = juntar_temas_descripciones(temas,descripciones)
                    with st.spinner("Trabajando en el comentario..."):
                        comentario = B._comment_from_ia(
                            B.TEMPLATE_COMMENT_PREGUNTAS,
                            temas_descripciones,
                            resumen,
                            llm_comentario,
                            st.session_state.get("autor",""))
                    st.session_state["outputs"]["Comentario"] = comentario

                # Metemos todo en tuplas y ordenamos por marca
                lista_todo_junto:list = sorted([(m,t,d) for m,t,d in zip(
                    marcas,
                    temas,
                    descripciones,
                )],
                key=lambda x:x[0])
                # metemos todo junto en sesion
                st.session_state["outputs"]["todo_junto"] = lista_todo_junto 
                st.rerun()                

    mostrar_outputs_titulo()    
    mostrar_outputs_resumen()
    mostrar_outputs_temas(descripcion_temas)
    # TODO agregar boton para descargar en word o txt a modo de seguridad?
    # Comentarios en youtube
    # Si las credenciales existen y son válidas
    if json_file_bytes and st.session_state["outputs"].get("todo_junto",""):
        # Botón Publicar temas
        comentar_yt_temas = st.button("Publicar temas", help="❗ Atención! Se escribirá un comentario en el video de youtube.")
        # instanciamos el gestor
        gestor_yt = ym.GestorYoutube(st.session_state["credenciales_json"])
        if comentar_yt_temas:
                        #Comprobamos que no se haya comentado ya antes para esa url
                        if not st.session_state[url].get("publicar_temas"):               
                            #creamos el string con las marcas, temas y descripciones
                            comentario_temas = crear_comentario_yt_temas(st.session_state["outputs"]["todo_junto"])
                            #Comentamos en youtube
                            gestor_yt.comentar(url,comentario_temas)
                            #Aumentamos contador
                            st.session_state[url]["publicar_temas"] = True
                        else:
                            st.error("Ya has comentado los temas en este video.")
    mostrar_comentario()
    # Comentarios en youtube
    if json_file_bytes and validar_credenciales:
        if comentario:=st.session_state["outputs"].get("Comentario",""):
                        # Botón Publicar comentario
                        comentar_yt_comentario = st.button("Publicar comentario", help="❗ Atención! Se escribirá un comentario en el video de youtube.")
                        if comentar_yt_comentario:
                            # Comprobamos que no se haya mandado ya el comentario
                            if not st.session_state[url].get("publicar_comentario"): 
                                # Comentamos en youtube
                                gestor_yt.comentar(url,comentario)
                                # Aumentamos contador
                                st.session_state[url]["publicar_comentario"] = True

                            else:
                                # Lanzamos error de que ya se ha comentado
                                st.error("Ya has comentado en este video.")

    st.session_state["activador_stream"] = False
    st.caption("Done by STM w/ 💗")

    with st.sidebar:
        st.header(f"{HEADER_EMOJI} 💶 Costes {HEADER_EMOJI}")
        st.subheader(f"{SUBHEADER_EMOJI} Costes estimados {SUBHEADER_EMOJI}")
        st.write("Tokens estimados",st.session_state.get("tokens_estimados",0))
        st.write("Coste estimado €",st.session_state.get("coste_estimado",0))
        
        if validada_url := st.session_state.get("url_valido",False):
            calcular_estimados = st.button("Estimar costes")

            if calcular_estimados:
                st.session_state["tokens_estimados"], st.session_state["coste_estimado"] = B.estimar_coste_generacion(
                    [B.TEMPLATE_TITULO,B.TEMPLATE_RESUMEN,B.TEMPLATE_TEMAS,B.TEMPLATE_COMMENT_PREGUNTAS],
                    [llm_titulo,llm_resumen,llm_temas,llm_comentario],
                    hay_titulo=mostrar_titulo,
                    hay_comentario=comentarios_auto,
                    transcripcion_tokens=st.session_state.get("total_tokens",0)
                )
                st.rerun()

        st.subheader(f"{SUBHEADER_EMOJI} Costes reales {SUBHEADER_EMOJI}")
        st.write("Tokens totales",st.session_state.get("total_tokens",0))
        st.write("Coste total €",round(st.session_state.get("total_cost",0),5))

st.session_state