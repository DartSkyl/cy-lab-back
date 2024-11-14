from langchain_community.llms import YandexGPT
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import asyncio

from configuration import MODEL_URI, FOLDER_ID


load_dotenv()

# Задаем модель чата
chat_model = YandexGPT(model_uri=MODEL_URI)

# Формируем векторную базу данных на основе текстового документа
loader = TextLoader(file_path='context_for_ai.txt', encoding='utf-8')
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
document = loader.load_and_split(text_splitter=splitter)
embedding = YandexGPTEmbeddings(folder_id=FOLDER_ID)
vector_store = FAISS.from_documents(document, embedding=embedding)
store = vector_store.as_retriever(search_kwargs={'k': 1})
chat_history = []

prompt_text = """
Ты клиент-менеджер компании занимающейся разработкой решений для бизнеса на основе ИИ. Твоя задача выяснить потребность 
клиента и узнать какую пользу для своего бизнеса хочет получить клиент. Для этого тебе нужно выяснить какой у клиента 
бизнес и в какой процесс он хочет интегрировать ИИ. К тебе будут обращаться владельцы бизнеса и
руководящий состав. Отвечай просто и кратко. Если клиент не знает что он хочет, то предложи ему несколько 
вариантов из тех, что описаны в контексте {context}. Веди диалог так, как будто ты человек. Ответы должны быть в формате 
обычных предложений. Длинна ответа должна быть не более 300 символов.
"""
prompt = ChatPromptTemplate.from_messages([
    ('system', prompt_text),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])
# Формируем цепочку
chain = create_stuff_documents_chain(llm=chat_model, prompt=prompt)
retrieval_chain = create_retrieval_chain(store, chain)


# Делаем запрос
async def process_chat(user_input, chat_history_list):

    response = await retrieval_chain.ainvoke({
        'input': user_input,
        # История чата отключается через пустой список
        'chat_history': chat_history_list,
    })
    return response['answer']


async def start_up():
    while True:
        user = input('You: ')
        if user != 'exit':
            res = await process_chat(user, chat_history)
            print(res)
            chat_history.append(HumanMessage(content=user))
            chat_history.append(AIMessage(content=res))
        else:
            break


if __name__ == '__main__':
    asyncio.run(start_up())
