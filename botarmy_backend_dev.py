import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
import json
from datetime import datetime

if 'train' not in st.session_state:
    st.session_state.train = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

def get_pdf_text(pdf_list):
    text = ""
    for pdf in pdf_list:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.warning("Please upload a textual PDF file - this PDF file contains images only.")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=st.session_state.api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Clear previous messages before showing new messages
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            col1, col2 = st.columns([1, 12])
            with col1:
                st.image('user_icon.png', width=50)
            with col2:
                st.write(msg.content)
        else:
            col1, col2 = st.columns([1, 12])
            with col1:
                st.image('bot_icon.png', width=50)
            with col2:
                st.write(msg.content)

def capture_user_data():
    st.title("Datos del usuario")
    with st.form(key="user_data_form"):
        name = st.text_input("Nombre completo")
        role = st.text_input("Rol en la empresa")
        submit_button = st.form_submit_button("Enviar")
        
        if submit_button and name and role:
            st.session_state.user_data = {"name": name, "role": role}
            st.success("Datos guardados correctamente")

def save_feedback(feedback):
    # Guardar el feedback en un archivo JSON
    user_data = st.session_state.user_data
    feedback_data = {
        "id": 1,  # ID inicial por defecto
        "name": user_data["name"],
        "role": user_data["role"],
        "feedback": feedback,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Fecha y hora actuales
    }

    # Verificar si el archivo feedbacks.json ya existe
    if os.path.exists("feedbacks.json"):
        try:
            with open("feedbacks.json", "r") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            # Si el archivo está vacío o tiene un formato incorrecto, iniciamos con una lista vacía
            data = []
    else:
        data = []

    # Asignar el ID como el siguiente número en la secuencia
    if data:
        feedback_data["id"] = data[-1]["id"] + 1  # Incrementar el ID basándose en el último valor

    # Agregar el nuevo feedback
    data.append(feedback_data)

    # Guardar en el archivo JSON
    with open("feedbacks.json", "w") as file:
        json.dump(data, file, indent=4)
    
    st.session_state.feedback_given = True
    st.success("Feedback enviado correctamente")

def feedback_form():
    st.title("Proporcione su feedback")
    feedback = st.text_area("Escriba su feedback sobre la capacitación recibida")
    if st.button("Enviar feedback"):
        if feedback:
            save_feedback(feedback)

def homepage():
    st.title("Bienvenidos a la plataforma de asistentes internos de :red[La Fabril]")
    st.markdown("En esta plataforma los trabajadores de la Fabril podrán interactuar y expresarse con la empresa.")
    st.warning("1. Introduzca la OPENAI API KEY cuando se la pidan.")
    st.success("2. Elija el asistente con el que quiera interactuar.")
    st.warning("3. Entrene al asistente pulsando el botón de entrenamiento.")
    st.success("4. Proporcione su feedback una vez que haya interactuado con el asistente.")
    st.success("5. Después de enviar su feedback, seleccione :red[Nueva Conversación] en la barra de navegación.")

def framework_page(framework, sample_pdf):
    st.title(f"Página de asistencia del {framework}")
    
    if st.session_state.user_data is None:
        capture_user_data()  # Capturar los datos del usuario antes de entrenar al asistente
    else:
        use_sample_pdf = st.checkbox(f"Interactua con el asistente del {framework}")
        if use_sample_pdf:
            st.session_state.pdf_files = [sample_pdf]
        
        train = st.button("Entrena al agente")
        if train:
            with st.spinner("Procesando"):
                raw_text = get_pdf_text(st.session_state.pdf_files)
                st.session_state.pdf_text = raw_text
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                if vector_store:
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.session_state.train = True

        if not st.session_state.train:
            st.warning("Primero entrene al agente")

        if st.session_state.train:
            st.write("<h5><br>Info acerca de la capacitación recibida dependiendo de su rol</h5>", unsafe_allow_html=True)
            user_question = st.text_input(label="", placeholder="Inserte su texto aquí...")
            if user_question:
                handle_user_input(user_question)

            if st.session_state.feedback_given:
                st.info("Gracias por su feedback.")
            else:
                feedback_form()  # Formulario de feedback después de interactuar con el asistente

def clear_conversation_page():
    st.title("Nueva conversación")
    st.info("Elija un asistente desde la barra de navegación lateral.")
    st.session_state.train = False
    st.session_state.feedback_given = False
    st.session_state.user_data = None

# Sidebar actions
st.sidebar.image('logo_la_fabril.jpeg', width=150)
st.sidebar.success('LA FABRIL')

api_key = st.sidebar.text_input("Inserte su OpenAI API Key", type="password")
if api_key:
    st.session_state.api_key = api_key

bot_type = st.sidebar.radio("Elija un agente", 
                            ["Inicio", "Equipo soporte IT", "Equipo soporte SAP", "Equipo soporte Redes", "Equipo soporte servidores", 
                              "Nueva conversación"])

if bot_type == "Inicio":
    homepage()
elif bot_type == "Equipo soporte IT":
    framework_page("Equipo soporte IT", os.path.join(os.getcwd(), "base1.pdf"))
elif bot_type == "Equipo soporte SAP":
    framework_page("Equipo soporte SAP", os.path.join(os.getcwd(), "base2.pdf"))
elif bot_type == "Equipo soporte Redes":
    framework_page("Equipo soporte Redes", os.path.join(os.getcwd(), "base3.pdf"))
elif bot_type == "Equipo soporte servidores":
    framework_page("Equipo soporte servidores", os.path.join(os.getcwd(), "base4.pdf"))
elif bot_type == "Nueva conversación":
    clear_conversation_page()
