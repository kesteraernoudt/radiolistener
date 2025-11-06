import asyncio
import json
from datetime import datetime
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
    
    @staticmethod
    def parse_datetime(datetime_str):
        """
        Parse datetime string in various formats:
        - "2025-11-04 10:30:00"
        - "2025-11-04 10:30"
        - "2025-11-04"
        - "10:30:00" (assumes today)
        - "10:30" (assumes today)
        """
        if not datetime_str:
            return None
        
        datetime_str = datetime_str.strip()
        today = datetime.now().date()
        
        # Try different formats
        formats = [
            "%Y-%m-%d %H:%M:%S",  # 2025-11-04 10:30:00
            "%Y-%m-%d %H:%M",     # 2025-11-04 10:30
            "%Y-%m-%d",            # 2025-11-04
            "%H:%M:%S",            # 10:30:00 (today)
            "%H:%M",               # 10:30 (today)
        ]
        
        for fmt in formats:
            try:
                if fmt.startswith("%H"):
                    # Time-only format - combine with today's date
                    parsed_time = datetime.strptime(datetime_str, fmt).time()
                    return datetime.combine(today, parsed_time)
                else:
                    parsed = datetime.strptime(datetime_str, fmt)
                    if fmt == "%Y-%m-%d":
                        # Date only - set to start of day
                        return parsed.replace(hour=0, minute=0, second=0, microsecond=0)
                    return parsed
            except ValueError:
                continue
        
        return None

    def bot_main(self):
        # Build the Application inside the async function to ensure it's in the correct event loop
        self.app = Application.builder().token(self.token).build()
        self.loop = asyncio.get_event_loop()
        self.app.add_handler(CommandHandler('start', self.start_command))
        self.app.add_handler(CommandHandler(['log','l'], self.log_command))
        self.app.add_handler(CommandHandler(['ailog','ail'], self.ailog_command))
        self.app.add_handler(CommandHandler(['text','t'], self.text_command))
        self.app.add_handler(CommandHandler(['ai','a'], self.ai_command))
        self.app.add_handler(CommandHandler(['radios','radio'], self.radios_command))
        self.app.add_handler(CommandHandler(['restart','r'], self.restart_command))
        self.app.add_handler(CallbackQueryHandler(self.button))
        self.app.add_handler(CommandHandler(['clip','c'], self.clip_command))
        self.app.add_handler(CommandHandler(['stats','s'], self.stats_command))
        self.app.add_handler(CommandHandler(['listcommands', 'list'], self.list_commands))
        self.app.add_handler(CommandHandler(['words', 'w'], self.list_codewords))
        self.app.add_handler(CommandHandler(['search', 'find', 'f'], self.search_command))
        self.app.run_polling(drop_pending_updates=True)

    async def list_commands(self, update, context):
        commands_list = []
        # Access handlers in the dispatcher
        for handler_group in context.application.handlers.values():
            for handler in handler_group:
                if isinstance(handler, CommandHandler):
                    commands_list.extend(handler.commands)

        if commands_list:
            await update.message.reply_text(f"Supported commands: {', '.join(['/' + cmd for cmd in commands_list])}")
        else:
            await update.message.reply_text("No commands defined.")

    async def start_command(self, update, context):
        await update.message.reply_text('Hello! I am your bot.')

    async def log_command(self, update, context):
        """Get log entries. Usage: /log [num_lines] [radio] [start_datetime]
        Examples:
        /log 50 Mix106.5
        /log 50 Mix106.5 2025-11-04 10:30:00
        /log 50 Mix106.5 2025-11-04
        /log 50 Mix106.5 10:30:00
        """
        num_lines = 10
        radio = ""
        start_datetime = None
        arg = 0
        
        if len(context.args) > arg and context.args[arg].isdigit():
            num_lines = int(context.args[arg])
            arg += 1
        
        # Parse radio name (if present)
        if len(context.args) > arg:
            next_arg = context.args[arg]
            # Check if it matches a radio name
            for radio_name in self.radioListener.controllers.keys():
                if radio_name.startswith(next_arg.upper()):
                    radio = next_arg
                    arg += 1
                    break
        
        # Parse date/time from remaining args (could be multiple words like "2025-11-04 10:30:00")
        if len(context.args) > arg:
            datetime_str = " ".join(context.args[arg:])
            start_datetime = self.parse_datetime(datetime_str)
        
        log_lines = logger.get_radio_log(radio, num_lines, start_datetime)
        if not log_lines:
            await update.message.reply_text("No logs found.")
            return
        
        # Only invert if no start_datetime was provided (when start_datetime is provided, 
        # results are already in chronological order from oldest to newest)
        if start_datetime is None:
            #invert the results so the most recent is last
            log_lines.reverse()

        # Format results - Telegram has a 4096 character limit per message
        msg_lines = []
        for line in log_lines:
            msg_lines.append(line)
            # Check message length (leave some buffer)
            msg = "\n".join(msg_lines)
            if len(msg) > 4000:
                # Send current batch and continue with remaining results
                msg_lines.pop()  # Remove the line that would exceed limit
                await update.message.reply_text("\n".join(msg_lines))
                msg_lines = [line]  # Start new message with the line we removed
        
        # Send remaining results
        if msg_lines:
            await update.message.reply_text("\n".join(msg_lines))

    async def ailog_command(self, update, context):
        """Get AI log entries. Usage: /ailog [num_lines] [radio] [start_datetime]
        Examples:
        /ailog 50 Mix106.5
        /ailog 50 Mix106.5 2025-11-04 10:30:00
        /ailog 50 Mix106.5 2025-11-04
        /ailog 50 Mix106.5 10:30:00
        """
        num_lines = 10
        radio = ""
        start_datetime = None
        arg = 0
        
        if len(context.args) > arg and context.args[arg].isdigit():
            num_lines = int(context.args[arg])
            arg += 1
        
        # Parse radio name (if present)
        if len(context.args) > arg:
            next_arg = context.args[arg]
            # Check if it matches a radio name
            for radio_name in self.radioListener.controllers.keys():
                if radio_name.startswith(next_arg.upper()):
                    radio = next_arg
                    arg += 1
                    break
        
        # Parse date/time from remaining args (could be multiple words like "2025-11-04 10:30:00")
        if len(context.args) > arg:
            datetime_str = " ".join(context.args[arg:])
            start_datetime = self.parse_datetime(datetime_str)
        
        log_lines = logger.get_radio_ai_log(radio, num_lines, start_datetime)
        if not log_lines:
            await update.message.reply_text("No AI logs found.")
            return
        
        # Only invert if no start_datetime was provided (when start_datetime is provided, 
        # results are already in chronological order from oldest to newest)
        if start_datetime is None:
            #invert the results so the most recent is last
            log_lines.reverse()

        # Format results - Telegram has a 4096 character limit per message
        msg_lines = []
        for line in log_lines:
            msg_lines.append(line)
            # Check message length (leave some buffer)
            msg = "\n".join(msg_lines)
            if len(msg) > 4000:
                # Send current batch and continue with remaining results
                msg_lines.pop()  # Remove the line that would exceed limit
                await update.message.reply_text("\n".join(msg_lines))
                msg_lines = [line]  # Start new message with the line we removed
        
        # Send remaining results
        if msg_lines:
            await update.message.reply_text("\n".join(msg_lines))

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
        text_lines = controller.processor.previous_texts[-num_lines:]
        if not text_lines:
            await update.message.reply_text("No text found.")
            return
        
        # Format results - Telegram has a 4096 character limit per message
        msg_lines = []
        for line in text_lines:
            msg_lines.append(line)
            # Check message length (leave some buffer)
            msg = "\n".join(msg_lines)
            if len(msg) > 4000:
                # Send current batch and continue with remaining results
                msg_lines.pop()  # Remove the line that would exceed limit
                await update.message.reply_text("\n".join(msg_lines))
                msg_lines = [line]  # Start new message with the line we removed
        
        # Send remaining results
        if msg_lines:
            await update.message.reply_text("\n".join(msg_lines))

    async def list_codewords(self, update, context):
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
        msg = "\n".join(controller.processor.previous_codewords[-num_lines:])
        if msg:
            await update.message.reply_text(f"Codewords for {radio}:\n{msg}")
        else:
            await update.message.reply_text(f"No codewords found for {radio}.")

    async def search_command(self, update, context):
        """Search logs for a keyword or phrase.
        
        Usage:
        /search keyword              -> search all radios for keyword
        /search keyword radio        -> search specific radio for keyword
        /search 50 keyword radio     -> search with max 50 results for specific radio
        """
        if not context.args:
            await update.message.reply_text("Usage: /search [max_results] <keyword> [radio]\nExample: /search emergency Mix106.5")
            return
        
        max_results = 50
        radio = ""
        keyword = ""
        arg = 0
        
        # Parse first arg as number if present
        if len(context.args) > arg and context.args[arg].isdigit():
            max_results = int(context.args[arg])
            arg += 1
        
        # The keyword is everything after the optional number, except the last arg if it matches a radio name
        # If only one word remains, it's always the keyword (even if it matches a radio name)
        if len(context.args) > arg:
            if len(context.args) == arg + 1:
                # Only one word remaining - it's the keyword
                keyword = context.args[arg]
            else:
                # Multiple words - check if last arg matches a radio name
                last_arg = context.args[-1]
                matching_radio = None
                for radio_name in self.radioListener.controllers.keys():
                    if radio_name.startswith(last_arg.upper()):
                        matching_radio = last_arg
                        break
                
                if matching_radio:
                    # Last arg is a radio name, everything before it is the keyword
                    radio = matching_radio
                    keyword = " ".join(context.args[arg:-1])
                else:
                    # No radio specified, everything after optional number is the keyword
                    keyword = " ".join(context.args[arg:])
        
        if not keyword:
            await update.message.reply_text("Please provide a keyword to search for.\nUsage: /search [max_results] <keyword> [radio]")
            return
        
        # Search the logs
        results = logger.search_radio_log(radio=radio, keyword=keyword, max_results=max_results)
        
        if not results:
            radio_msg = f" for {radio}" if radio else ""
            await update.message.reply_text(f"No matches found for '{keyword}'{radio_msg}")
            return
        
        #invert the results so the most recent is last
        results.reverse()

        # Format results - Telegram has a 4096 character limit per message
        msg_lines = []
        for line in results:
            msg_lines.append(line)
            # Check message length (leave some buffer)
            msg = "\n".join(msg_lines)
            if len(msg) > 4000:
                # Send current batch and continue with remaining results
                msg_lines.pop()  # Remove the line that would exceed limit
                await update.message.reply_text("\n".join(msg_lines))
                msg_lines = [line]  # Start new message with the line we removed
        
        # Send remaining results
        if msg_lines:
            final_msg = "\n".join(msg_lines)
            if len(results) >= max_results:
                final_msg += f"\n\n(Showing first {max_results} results)"
            await update.message.reply_text(final_msg)

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

    async def stats_command(self, update, context):
        """Dump stats
        """
        radio = ""
        arg = 0
        if len(context.args) > arg:
            radio = context.args[arg]
        if radio:
            controller = self.radioListener.controller(radio)
            if controller is None or controller.processor is None:
                await update.message.reply_text(f"No such radio station found ({radio}) or processor not initialized.")
                return
            stats = controller.get_stats()
            await update.message.reply_text(f"{controller.RADIO_CONF.get('NAME','UNKNOWN')}:\n{json.dumps(stats, indent=2)}")
        else:
            stats = {}
            for controller in self.radioListener.controllers.values():
                stats[controller.RADIO_CONF.get('NAME','UNKNOWN')] = controller.get_stats()
            await update.message.reply_text(json.dumps(stats, indent=2))

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