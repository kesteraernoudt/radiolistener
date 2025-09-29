import asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from utils import logger
from audio import clip_saver

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
        self.app.add_handler(CommandHandler(['log','l'], self.log_command))
        self.app.add_handler(CommandHandler(['text','t'], self.text_command))
        self.app.add_handler(CommandHandler(['ai','a'], self.ai_command))
        self.app.add_handler(CommandHandler(['radios','radio'], self.radios_command))
        self.app.add_handler(CommandHandler(['restart','r'], self.restart_command))
        self.app.add_handler(CallbackQueryHandler(self.button))
        self.app.add_handler(CommandHandler(['clip','c'], self.clip_command))
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

    async def restart_command(self, update, context):
        radio = ""
        if len(context.args) > 0:
            radio = context.args[0]
        controller = self.radioListener.controller(radio)
        if controller is None:
            await update.message.reply_text(f"No such radio station found ({radio})")
            return
        controller.restart()
        await update.message.reply_text(f"Restarted {controller.RADIO_CONF.get('NAME','UNKNOWN')}")

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

    async def clip_command(self, update, context):
        """Save current audio buffer and send context + audio to the invoking chat.

        Usage examples:
        /clip            -> save clip for default radio (first available) and send last CONTEXT_LEN lines
        /clip 5          -> save clip and send last 5 lines
        /clip station    -> save clip for station (prefix match)
        /clip 4 station  -> save clip for station with 4 context lines
        """
        num_lines = None
        radio = ""
        arg = 0
        # parse first arg as number if present
        if len(context.args) > arg and context.args[arg].isdigit():
            num_lines = int(context.args[arg])
            arg += 1
        if len(context.args) > arg:
            radio = context.args[arg]

        controller = self.radioListener.controller(radio)
        if controller is None or controller.processor is None:
            await update.message.reply_text(f"No such radio station found ({radio}) or processor not initialized.")
            return

        processor = controller.processor
        if num_lines is None:
            # default to processor CONTEXT_LEN if available, else 3
            num_lines = getattr(processor, 'CONTEXT_LEN', 3)

        # snapshot rolling buffer and previous texts
        with processor.lock:
            audio_bytes = bytes(processor.rolling_buffer) if processor.rolling_buffer else b""
            context_slice = list(processor.previous_texts[-num_lines:]) if processor.previous_texts else []
        context_text = "\n".join(context_slice) if context_slice else ""

        if not audio_bytes:
            await update.message.reply_text("No audio in buffer to save.")
            return

        try:
            filename = clip_saver.save_clip(audio_bytes)
        except Exception as e:
            await update.message.reply_text(f"Failed to save clip: {e}")
            return

        # build context text
        context_text = "\n".join(processor.previous_texts[-num_lines:]) if processor.previous_texts else ""
        header = f"Clip saved for {controller.RADIO_CONF.get('NAME','UNKNOWN')}\n"
        if context_text:
            await update.message.reply_text(header + "Context:\n" + context_text)
        else:
            await update.message.reply_text(header + "(no recent speech captured)")

        # send audio file back to invoking chat
        try:
            # use reply_audio if available
            with open(filename, 'rb') as af:
                await update.message.reply_audio(audio=af, caption=f"Saved clip from {controller.RADIO_CONF.get('NAME','UNKNOWN')}")
        except Exception as e:
            await update.message.reply_text(f"Clip saved to {filename} but failed to send audio: {e}")

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
        text = text.replace(" ", "%20")
        sms_url = f"sms:{phone_number}&body={text}"
        keyboard = [ [InlineKeyboardButton(f"SMS {text} to {phone_number}", callback_data=sms_url)] ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        #msg=f"sms:{phone_number}?body={text}"
        asyncio.run_coroutine_threadsafe(self.app.bot.send_message(chat_id=self.radioListener.CONFIG["TELEGRAM_CHAT_ID"], text="Send SMS?", reply_markup=reply_markup), self.loop)

    @staticmethod
    async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Parses the CallbackQuery and updates the message text."""
        query = update.callback_query

        # CallbackQueries need to be answered, even if no notification to the user is needed
        # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
        await query.answer()
        
        await query.edit_message_text(text=f"{query.data}")