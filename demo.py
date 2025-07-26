import streamlit as st 
import google.generativeai as genai 
import os 
from dotenv import load_dotenv 


# Tải biến môi trường 
load_dotenv() 

# Truy cập lấy key từ .env 
api_key = os.getenv("GEMINI_API_KEY")

# Khởi tạo môi trường tương tác với Gemini API 
genai.configure(api_key=api_key) 

# Thiết lập cấu hình chương trình 
st.set_page_config(page_title="Gemini Clone", layout="wide")

st.title("🧠 Gemini Clone Chatbot") 

def generate_bot_response(question):
    try:
        # Truy cập vào mô hình 
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Xử lý câu hỏi của user 
        response = model.generate_content(question) 
        
        # Xử lý câu trả lời 
        bot_response = response.text
        
        return bot_response 
    except Exception as e:
        return f"Đã xảy ra lỗi: {str(e)}" 
    
# Tạo một text input cho người dùng nhập câu hỏi 
prompt = st.chat_input("Nhập câu hỏi của bạn ...")

if prompt: 
    # Hiển thị câu hỏi người dùng 
    with st.chat_message("user"):
        st.markdown(prompt)
    
    response = generate_bot_response(prompt) 
    
    # Hiển thị phản hồi từ Gemini bot:
    with st.chat_message("assistant"):
        st.markdown(response)