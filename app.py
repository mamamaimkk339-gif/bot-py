import asyncio
import logging
import os
import gc
import openai
from openai import AsyncOpenAI
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

# Ключи берутся из переменных окружения (задаются в настройках Infrlo)
BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

SYSTEM_PROMPT = "Ты дружелюбный и полезный ИИ-ассистент. Отвечай на языке пользователя."
MAX_HISTORY = 10  # Максимум сообщений в истории (для экономии памяти)
MAX_USERS = 50    # Максимум пользователей, чью историю храним одновременно

# Актуальные и стабильные модели на Groq (август 2026)
# Если 404 - код переключится на следующую
MODELS = [
    "llama-3.3-70b-versatile", 
    "gemma2-9b-it", 
    "mixtral-8x7b-32768"
]

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

histories = {}

def get_history(user_id):
    """Безопасно получает историю, удаляя старых пользователей при переполнении."""
    if user_id not in histories:
        if len(histories) >= MAX_USERS:
            histories.pop(next(iter(histories))) # Удаляем самого старого
        histories[user_id] = []
    return histories[user_id]

async def ask_groq(history):
    last_error = None
    for model in MODELS:
        for attempt in range(2):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
                )
                return response.choices[0].message.content or "(пустой ответ)"
            
            except openai.NotFoundError:
                # Ошибка 404: Модель удалена или недоступна. Пробуем следующую.
                logging.warning(f"Модель {model} не найдена, переключаюсь...")
                break 
            
            except (openai.RateLimitError, openai.InternalServerError):
                # Ошибка 429 (лимиты) или 500/502 (сервер Groq перегружен). Ждем.
                logging.warning(f"Модель {model} перегружена или лимит. Ждем...")
                await asyncio.sleep(2)
                last_error = "Превышен лимит запросов. Попробуйте позже."
                continue
            
            except openai.APIConnectionError:
                last_error = "Ошибка сети. Проверьте подключение."
                await asyncio.sleep(1)
                continue
                
            except Exception as e:
                # Другие ошибки (например, неправильный API ключ)
                last_error = e
                break
                
    raise Exception(last_error if last_error else "Не удалось получить ответ от ИИ.")

@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Привет! 👋 Я ИИ-бот. Напиши мне любой вопрос.\n/reset — очистить историю диалога.")

@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    if message.from_user.id in histories:
        histories.pop(message.from_user.id)
    gc.collect()
    await message.answer("История очищена ✅")

@dp.message(F.text)
async def chat(message: Message):
    history = get_history(message.from_user.id)
    
    # Добавляем сообщение пользователя
    history.append({"role": "user", "content": message.text})
    
    # Обрезаем историю, если она стала слишком длинной
    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]

    await bot.send_chat_action(message.chat.id, "typing")

    try:
        answer = await ask_groq(history)
    except Exception as e:
        logging.error(f"Ошибка при запросе к Groq: {e}")
        history.pop() # Убираем сообщение пользователя, так как ответа не было
        await message.answer(f"⚠️ Ошибка: {e}")
        return

    # Добавляем ответ бота в историю
    history.append({"role": "assistant", "content": answer})
    
    # Отправляем ответ (разбивая на части, если он слишком длинный)
    for i in range(0, len(answer), 4096):
        await message.answer(answer[i:i + 4096])
        
    gc.collect() # Очистка памяти

async def main():
    # Ограничиваем количество одновременных задач для экономии RAM
    await dp.start_polling(bot, tasks_concurrency_limit=10)

if __name__ == "__main__":
    asyncio.run(main())
