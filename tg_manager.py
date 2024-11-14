from pyrogram import Client
from pyrogram.filters import incoming
from pyrogram.types import Message
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_core.messages import HumanMessage, AIMessage

import logging

from ai_core import process_chat

logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.DEBUG)


app = Client('cy_lab_ai')
clients_dict = dict()


async def check_chat(chat_id, user_msg, client: Client):
    ai_answer = await process_chat(user_msg, clients_dict[chat_id]['chat_history'])
    await client.send_message(chat_id, ai_answer)
    clients_dict[chat_id]['chat_history'].append(HumanMessage(content=user_msg))
    clients_dict[chat_id]['chat_history'].append(AIMessage(content=ai_answer))
    # await timer.remove_timer(chat_id)


# class AnswerTimer:
#     def __init__(self):
#         self._scheduler = AsyncIOScheduler()
#         self._scheduler.start()
#
#     async def start_answer_timer(self, chat_id, user_msg, client):
#         self._scheduler.add_job(
#             func=check_chat,
#             kwargs={'chat_id': chat_id, 'user_msg': user_msg, 'client': client},
#             id=str(chat_id),
#             trigger='interval',
#             seconds=1,
#             max_instances=1,
#             replace_existing=True
#         )
#
#     async def remove_timer(self, chat_id):
#         self._scheduler.remove_job(str(chat_id))
#
#
# timer = AnswerTimer()


@app.on_message(filters=incoming)
async def ai_answer_to_client(client: Client, message: Message):
    """Ловим сообщение того кто нам написал и если хозяин долго не отвечает, то отвечаем за него"""
    # await client.send_message(message.from_user.id, 'AI answer 😘')
    if not clients_dict.get(message.from_user.id):
        clients_dict[message.from_user.id] = {
            'chat_history': []
        }
    await check_chat(message.from_user.id, message.text, client)


app.run()
