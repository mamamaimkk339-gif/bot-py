import asyncio
import logging
import os
import gc
from openai import AsyncOpenAI
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

# Ключи берутся из переменных окружения
BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

SYSTEM_PROMPT = "Ты дружелюбный и полезный ИИ-ассистент. Отвечай на языке пользователя."
MAX_HISTORY = 10 # Уменьшено с 20 до 10, чтобы экономить память
MAX_USERS = 50   # Максимальное количество пользователей, чью историю храним

# Жестко заданные модели (самые быстрые и стабильные у Groq)
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# Используем обычный dict вместо defaultdict
histories = {}

def get_history(user_id):
    """Безопасно получает историю, удаляя старых пользователей при переполнении."""
    if user_id not in histories:
        if len(histories) >= MAX_USERS:
            # Удаляем самого старого пользователя (первый ключ в словаре)
            histories.pop(next(iter(histories)))
        histories[user_id] = []
    return histories[user_id]

async def ask_groq(history):
    last_error = None
    for model in MODELS:
        for attempt in range(2): # Уменьшено с 3 до 2 попыток
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
                )
                return response.choices[0].message.content or "(пустой ответ)"
            except Exception as e:
                last_error = e
                text = str(e)
                if "429" in text or "502" in text or "503" in text:
                    await asyncio.sleep(2)
                    continue
                if "404" in text or "400" in text:
                    break
                raise
    raise last_error

@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Привет! 👋 Я ИИ-бот на Groq. Напиши мне вопрос.\n/reset — очистить историю")

@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    if message.from_user.id in histories:
        histories.pop(message.from_user.id)
    gc.collect() # Очищаем память после сброса
    await message.answer("История очищена ✅")

@dp.message(F.text)
async def chat(message: Message):
    history = get_history(message.from_user.id)
    
    # Обрезаем историю, чтобы она не разрасталась
    history.append({"role": "user", "content": message.text})
    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]

    await bot.send_chat_action(message.chat.id, "typing")

    try:
        answer = await ask_groq(history)
    except Exception as e:
        logging.exception("Ошибка Groq")
        history.pop() # Убираем последнее сообщение, если была ошибка
        await message.answer(f"⚠️ Ошибка: {type(e).__name__}: {e}")
        return

    history.append({"role": "assistant", "content": answer})
    
    # Отправка длинных сообщений
    for i in range(0, len(answer), 4096):
        await message.answer(answer[i:i + 4096])
        
    # Принудительная очистка памяти после каждого ответа
    gc.collect()

async def main():
    # tasks_concurrency_limit=10 не даст боту создать слишком много задач сразу
    await dp.start_polling(bot, tasks_concurrency_limit=10)

if __name__ == "__main__":
    asyncio.run(main())
