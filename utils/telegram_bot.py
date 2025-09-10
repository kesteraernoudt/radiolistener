import asyncio
from telegram.ext import Application, CommandHandler
from utils import logger

class TelegramBot():
    def __init__(self, token, radioListener):
        self.token = token
        self.radioListener = radioListener
        self.app = None # Initialize Application here
        self.loop = None

    def bot_main(self):
        # Build the Application inside the async function to ensure it's in the correct event loop
        self.app = Application.builder().token(self.token).build()
        self.loop = asyncio.get_event_loop()
        self.app.add_handler(CommandHandler('start', self.start_command))
        self.app.add_handler(CommandHandler('log', self.log_command))
        self.app.add_handler(CommandHandler('text', self.text_command))
        self.app.add_handler(CommandHandler('ai', self.ai_command))
        self.app.add_handler(CommandHandler('radios', self.radios_command))
        
        self.app.run_polling()

    async def start_command(self, update, context):
        await update.message.reply_text('Hello! I am your bot.')

    async def log_command(self, update, context):
        num_lines = 10
        radio = ""
        arg = 0
        if len(context.args) > arg and context.args[arg].isdigit():
            num_lines = int(context.args[arg])
            arg += 1
        if len(context.args) > arg:
            radio = context.args[arg]
        msg = "\n".join(logger.get_radio_log(radio, num_lines))
        if msg:
            await update.message.reply_text(msg)

    async def radios_command(self, update, context):
        radios = "\n".join(self.radioListener.controllers.keys())
        if radios:
            await update.message.reply_text(radios)

    async def text_command(self, update, context):
        num_lines = 10
        radio = ""
        arg = 0
        if len(context.args) > arg and context.args[arg].isdigit():
            num_lines = int(context.args[arg])
            arg += 1
        if len(context.args) > arg:
            radio = context.args[arg]
        controller = self.radioListener.controller(radio)
        if controller is None or controller.processor is None:
            await update.message.reply_text(f"No such radio station found ({radio}) or processor not initialized.")
            return
        msg = "\n".join(controller.processor.previous_texts[-num_lines:])
        if msg:
            await update.message.reply_text(msg)

    async def ai_command(self, update, context):
        num_lines = 3
        radio = ""
        arg = 0
        if len(context.args) > arg and context.args[arg].isdigit():
            num_lines = int(context.args[arg])
            arg += 1
        if len(context.args) > arg:
            radio = context.args[arg]
        controller = self.radioListener.controller(radio)
        if controller is None or controller.processor is None:
            await update.message.reply_text(f"No such radio station found ({radio}) or processor not initialized.")
            return
        msg = "\n".join(controller.processor.previous_texts[-num_lines:])
        if msg:
            codeword = controller.processor.genAIHandler.generate(msg)
            await update.message.reply_text(codeword if codeword else "No codeword found")

    def send_message(self, text):
        if self.app is None:
            return
        asyncio.run_coroutine_threadsafe(self.app.bot.send_message(chat_id=self.radioListener.CONFIG["TELEGRAM_CHAT_ID"], text=text), self.loop)

    def send_audio(self, audio_path, caption=""):
        if self.app is None:
            return
        asyncio.run_coroutine_threadsafe(self.app.bot.send_audio(chat_id=self.radioListener.CONFIG["TELEGRAM_CHAT_ID"], audio=open(audio_path, 'rb'), caption=caption), self.loop)
    
    def send_sms_message(self, phone_number, text = ""):
        if not text:
            text = "codeword"
        text.replace(" ", "%20")
        msg = f"[send sms](sms:{phone_number}&body={text})"
        #msg=f"sms:{phone_number}?body={text}"
        asyncio.run_coroutine_threadsafe(self.app.bot.send_message(chat_id=self.radioListener.CONFIG["TELEGRAM_CHAT_ID"], text=msg, parse_mode='MarkdownV2'), self.loop)
