import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

#tải biến môi trường
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Khong tim thay API Key")
    st.stop()

genai.configure(api_key=api_key)

# heper
def get_pdf_text(pdf_docs):
    text = ""
    try: 
        for pdf in pdf_docs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf.read())
                tmp_file_path = tmp_file.name
            pdf_reader = PyPDFLoader(tmp_file_path)
            for page in pdf_reader.load_and_split():
                text += page.page_content
            os.unlink(tmp_file_path) # Xoa File Tam
    except Exception as e:
        st.error(f"Loi doc file PDF: {str(e)}")
    return text

def get_text_chunks(text): 
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text) 
        return chunks 
    except Exception as e: 
        st.error(f"Lỗi chia chunk: {str(e)}")
        return [] 
    

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embeddings) 
        vector_store.save_local("faiss_index")
        st.success("Tài liệu đã được phân tích xong, sẵn sàng để trả lời câu hỏi") 
    except Exception as e:
        st.error(f"Lỗi lưu vector database: {str(e)}")    
  
def get_conversational_chain():
    prompt_template = """
    Trả lời câu hỏi một cách chi tiết nhất có thể dựa trên ngữ cảnh được cung cấp. Nếu câu trả lời không có trong ngữ cảnh được cung cấp, hãy nói, "Câu trả lời không có trong ngữ cảnh."
    Không cung cấp thông tin sai lệch.

    Ngữ cảnh: {context}
    Câu hỏi: {question}

    Answer:
    """
    try: 
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        prompt =  PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)  
        return chain 
    except Exception as e: 
        st.error(f"Lỗi trong quá trình phân tích: {str(e)}")
        return None 
    
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if not os.path.exists("faiss_index"):
            st.error("Không tìm thấy FAISS index. Hãy tải tài liệu PDF lên trước.")
            return 
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
        docs = new_db.similarity_search(user_question) 
        chain = get_conversational_chain() 
        
        if not chain:
            return
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )        

        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Lỗi xử lý câu hỏi: {str(e)}")
   
# Set up trang chinh streamlit
st.set_page_config(page_title="Chat PDF RAG")
st.title("Chat bot phan tich trai lieu PDF")

user_question = st.text_input("Ban hay hoi sau khi tai lieu da duoc phan tich xong")

if user_question:
    user_input(user_question)

with st.sidebar:
    st.title("Menu")
    pdf_docs = st.file_uploader("Tai tai lieu PDF cua ban len", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Phân tích tài liệu"):
        if not pdf_docs:
            st.error("Vui lòng tải tài liệu lên trước") 
        with st.spinner("Đang xử lý..."): 
            raw_text = get_pdf_text(pdf_docs) 
            if raw_text: 
                text_chunks = get_text_chunks(raw_text) 
                if text_chunks:
                    get_vector_store(text_chunks)
                else:
                    st.error(" kieerm trai laij noi dung PDF")
                            
 
                
# Test tính năng
# if text_chunks:
#     st.write(text_chunks) 
# else:
#     st.error("Có lỗi") 