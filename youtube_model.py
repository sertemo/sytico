import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import json
import google.oauth2.service_account as gserac
import streamlit as st

class GestorYoutube:
    """Clase para gestionar credenciales de youtube, comentar y manejar excepciones
    """
    def __init__(self,file_json:dict):
        self.file_json = file_json

    def _get_authenticated_service(self):
        # Ruta al archivo de credenciales descargado
        client_secrets_file = self.file_json
    
        # Estos scopes permiten que tu aplicación lea y escriba comentarios
        scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
        
        # Obtén credenciales y devuelve el servicio autenticado
        try:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_config(client_secrets_file, scopes)
            credentials = flow.run_local_server()
        except OSError as oe:
            st.error(f"Se ha producido el siguiente error al publicar:\n{oe}\nReinicia la app.")
            st.stop()
        youtube = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)
        return youtube
    
    def _get_authenticated_service_bard(self):
        # Ruta al archivo de credenciales descargado
        client_secrets_file = self.file_json
    
        credentials = gserac.Credentials.from_service_account_file(client_secrets_file)
        #! DEBUG
        print(type(credentials), credentials)
        youtube = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)
        return youtube
    
    def comentar(self,video_url, comment_text):
        youtube = self._get_authenticated_service()
        video_id = video_url.split('v=')[-1].split('&')[0]
        try: 
            request = youtube.commentThreads().insert(
                part="snippet",
                body={
                    "snippet": {
                        "videoId": video_id,
                        "topLevelComment": {
                            "snippet": {
                                "textOriginal": comment_text
                            }
                        }
                    }
                }
            )
            response = request.execute()
            st.success(f"Se ha comentado exitosamente en el video: {video_url}")
        except googleapiclient.errors.HttpError as exc_http:
            st.error(f"Se ha producido el siguiente error al comentar: {exc_http}")
        except googleapiclient.errors.Error as exc_error:
            st.error(f"Se ha producido el siguiente error al comentar: {exc_error}")
        except Exception as exc:
            st.error(f"Se ha producido el siguiente error al comentar: {exc}")
        #!Para debug
        #print(json.dumps(response,indent=4,ensure_ascii=False))