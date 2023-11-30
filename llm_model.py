from langchain.chat_models import ChatOpenAI

class GestorModelos:
    """clase que se ocupa se gestionar los modelos llm adecuados
    para generar titulo resumen temas y comentario en funcion 
    de disponibilidad de gpt-4
    """
    def __init__(self,
                lista_nombre_modelos:list[str],
                acceso_gpt4:bool,
                openai_api_key:str
                ):
        self.api_key = openai_api_key
        self.lista_nombres_modelos =lista_nombre_modelos
        self.acceso_gpt4 = acceso_gpt4
        self.temperaturas = [0.4,0.4,0.2,0.7]
        self.GPT4_LARGE = "gpt-4" #No tengo acceso a gpt-4-32k
        self.GPT4 = "gpt-4"
        self.GPT3 = "gpt-3.5-turbo"
        self.GPT3_LARGE = "gpt-3.5-turbo-16k"
        
    #orden de los modelos: 
    # titulo, 
    # resumen, 
    # temas&stamps [MODELO LARGE CONTEXT], 
    # comentario

    def _get_modelo(self,model_name:str="gpt-3.5-turbo",temperatura:float=0)->ChatOpenAI:
        """Devuelve un modelo de clase OpenAI
        Si no se especifica el nombre devuelve el modelo mas basico

        Parameters
        ----------
        model_name : str, optional
            _description_, by default "gpt-3.5-turbo"
        temperatura : float, optional
            _description_, by default 0

        Returns
        -------
        ChatOpenAI
            _description_
        """
        return ChatOpenAI(
            temperature=temperatura,
            model_name=model_name,
            request_timeout = 180,
            openai_api_key=self.api_key
        )
    
    def _corregir_nombres_modelos(self)->None:
        """Función que corrige la los modelos elegidos en caso
        de no tener acceso a gpt-4
        """
        lista_nombres_buenos = []
        for idx,_ in enumerate(self.lista_nombres_modelos):
            #Si no tenemos acceso a gpt4. modificamos los modelos
            if not self.acceso_gpt4:
                if idx == 2:
                    lista_nombres_buenos.append(self.GPT3_LARGE)
                else:
                    lista_nombres_buenos.append(self.GPT3)                
                self.lista_nombres_modelos = lista_nombres_buenos
        #print(self.lista_nombres_modelos)
    
    def instanciar_modelos(self)->list[ChatOpenAI]:
        self._corregir_nombres_modelos()
        return [
            self._get_modelo(nombre,tempe) 
            for nombre, tempe 
            in zip(self.lista_nombres_modelos,self.temperaturas)
            ]