import os
import openai
import streamlit as st
import warnings
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI as LangChainOpenAI

warnings.filterwarnings("ignore")

# إعداد مفتاح API من بيئة Streamlit Cloud
openai.api_key = os.getenv("OPENAI_API_KEY")

# عنوان التطبيق
st.title("🎓 AlsunBot - Your School Assistant")

# مربع إدخال المستخدم
user_input = st.text_input("Ask me anything about Alsun International Schools:")

# تحميل بيانات المدرسة (مرة واحدة فقط)
@st.cache_resource
def load_index():
    loader = TextLoader("data.txt")
    data = loader.load()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)

    index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
    return index

index = load_index()

# إنشاء نموذج اللغة
llm = LangChainOpenAI(api_key=openai.api_key, temperature=0)

# الرد على المستخدم
if user_input:
    result = index.query(user_input, llm=llm, retriever_kwargs={"search_kwargs": {"k": 1}})
    st.write("**AlsunBot:**", result)

