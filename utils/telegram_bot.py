import asyncio
import json
from collections import deque
from io import BytesIO
from datetime import datetime
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, NetworkError, RetryAfter
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from utils import logger
from audio import clip_saver

class TelegramBot():
    MAX_MESSAGE_LENGTH = 4000
    DEBUG_HISTORY_LIMIT = 5

    def __init__(self, token, radioListener):
        self.token = token
        self.radioListener = radioListener
        self.app = None # Initialize Application here
        self.loop = None
        self.debug_history = deque(maxlen=self.DEBUG_HISTORY_LIMIT)
    
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

    async def _reply_text_or_document(self, update, content: str, filename: str, caption: str = ""):
        """Send text unless it exceeds Telegram limits, otherwise fall back to a document."""
        if len(content) <= self.MAX_MESSAGE_LENGTH:
            try:
                await update.message.reply_text(content)
                return
            except BadRequest as exc:
                if "Message is too long" not in str(exc):
                    raise
        bio = BytesIO(content.encode("utf-8"))
        bio.name = filename
        await update.message.reply_document(document=bio, caption=caption or filename)

    def bot_main(self):
        # Build the Application inside the async function to ensure it's in the correct event loop
        self.app = Application.builder().token(self.token).build()
        self.loop = asyncio.get_event_loop()
        self.app.add_error_handler(self.handle_telegram_error)
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
        self.app.add_handler(CommandHandler(['debug', 'debugprivate'], self.debug_command))
        self.app.add_handler(CommandHandler(['listcommands', 'list'], self.list_commands))
        self.app.add_handler(CommandHandler(['words', 'w'], self.list_codewords))
        self.app.add_handler(CommandHandler(['search', 'find', 'f'], self.search_command))
        self.app.run_polling(drop_pending_updates=True)

    def _log_background_exception(self, description, exc):
        msg = f"{description} failed: {exc}"
        logger.log_event("TELEGRAM", msg)
        self.debug_history.append(msg)

    def _schedule(self, coro, description):
        if self.app is None or self.loop is None or self.loop.is_closed():
            return
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        future.add_done_callback(lambda f: self._handle_future(f, description))

    def _handle_future(self, future, description):
        try:
            exc = future.exception()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._log_background_exception(description, exc)
            return
        if exc:
            self._log_background_exception(description, exc)

    async def _send_message(self, chat_id, text, reply_markup=None):
        max_attempts = 2
        attempt = 0
        while True:
            attempt += 1
            try:
                await self.app.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
                return
            except RetryAfter as exc:
                await asyncio.sleep(min(exc.retry_after, 30))
            except NetworkError:
                if attempt >= max_attempts:
                    raise
                await asyncio.sleep(1 + attempt)

    async def _send_audio(self, chat_id, audio_path, caption=""):
        max_attempts = 2
        attempt = 0
        while True:
            attempt += 1
            try:
                with open(audio_path, "rb") as audio_file:
                    await self.app.bot.send_audio(chat_id=chat_id, audio=audio_file, caption=caption)
                return
            except RetryAfter as exc:
                await asyncio.sleep(min(exc.retry_after, 30))
            except NetworkError:
                if attempt >= max_attempts:
                    raise
                await asyncio.sleep(1 + attempt)

    async def handle_telegram_error(self, update, context):
        error = context.error
        update_id = getattr(update, "update_id", None) if update else None
        chat_id = getattr(getattr(update, "effective_chat", None), "id", None)
        user_id = getattr(getattr(update, "effective_user", None), "id", None)
        details = []
        if update_id is not None:
            details.append(f"update_id={update_id}")
        if chat_id is not None:
            details.append(f"chat_id={chat_id}")
        if user_id is not None:
            details.append(f"user_id={user_id}")
        suffix = f" ({', '.join(details)})" if details else ""
        logger.log_event("TELEGRAM", f"Handler error: {error}{suffix}")
        self.debug_history.append(f"Handler error: {error}{suffix}")

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

    async def debug_command(self, update, context):
        """Toggle private debug notifications.
        Usage: /debug [on|off|status]
        """
        was_enabled = bool(self.radioListener.CONFIG.get("TELEGRAM_DEBUG_PRIVATE", False))
        enabled = was_enabled
        arg = context.args[0].lower() if context.args else "status"
        if arg in ("on", "enable", "enabled", "1", "true", "yes"):
            enabled = True
            self.radioListener.CONFIG["TELEGRAM_DEBUG_PRIVATE"] = True
        elif arg in ("off", "disable", "disabled", "0", "false", "no"):
            enabled = False
            self.radioListener.CONFIG["TELEGRAM_DEBUG_PRIVATE"] = False
        elif arg in ("status", "state"):
            pass
        else:
            await update.message.reply_text("Usage: /debug [on|off|status]")
            return

        private_chat_id = self.radioListener.CONFIG.get("TELEGRAM_CHAT_ID_PRIVATE")
        state = "enabled" if enabled else "disabled"
        if not private_chat_id:
            await update.message.reply_text(
                f"Private debug messages are {state}, but TELEGRAM_CHAT_ID_PRIVATE is not set."
            )
            return

        await update.message.reply_text(f"Private debug messages are {state}.")
        if enabled and not was_enabled and self.debug_history:
            for msg in list(self.debug_history):
                await context.application.bot.send_message(chat_id=private_chat_id, text=msg)

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
        if radio:
            controller = self.radioListener.controller(radio)
            if controller is None or controller.processor is None:
                await update.message.reply_text(
                    f"No such radio station found ({radio}) or processor not initialized."
                )
                return
            controllers = [controller]
        else:
            controllers = list(self.radioListener.controllers.values())
            if not controllers:
                await update.message.reply_text("No radio stations configured.")
                return

        lines = []
        if len(controllers) == 1:
            station_name = controllers[0].RADIO_CONF.get("NAME", "UNKNOWN")
            lines.append(f"Codewords for {station_name}:")
        else:
            lines.append(f"Recent codewords (last {num_lines} per station):")

        for ctrl in controllers:
            station_name = ctrl.RADIO_CONF.get("NAME", "UNKNOWN")
            processor = ctrl.processor
            entry_prefix = "  " if len(controllers) > 1 else ""
            if len(controllers) > 1:
                lines.append(f"{station_name}:")
            if processor is None:
                lines.append(f"{entry_prefix}(processor not initialized)")
                continue
            processor._clear_codewords_if_stale()
            with processor.lock:
                entries = list(processor.previous_codewords)
            if num_lines is not None:
                entries = entries[-num_lines:]
            if not entries:
                lines.append(f"{entry_prefix}(none)")
            else:
                for word, ts in entries:
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "unknown time"
                    lines.append(f"{entry_prefix}{ts_str} - {word}")
            if len(controllers) > 1:
                lines.append("")

        payload = "\n".join(lines).strip()
        if not payload:
            payload = "No codewords found."
        await self._reply_text_or_document(update, payload, "codewords.txt", caption="Codewords")

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
            codeword = controller.processor.genAIHandler.generate(msg, radio=controller.RADIO_CONF.get("NAME", ""))
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
            capture_ts = processor.segment_times[0][2] if getattr(processor, "segment_times", None) else None
        context_text = "\n".join(context_slice) if context_slice else ""

        if not audio_bytes:
            await update.message.reply_text("No audio in buffer to save.")
            return

        try:
            filename = clip_saver.save_clip(audio_bytes, capture_ts=capture_ts)
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
            station_name = controller.RADIO_CONF.get('NAME','UNKNOWN')
            payload = f"{station_name}:\n{json.dumps(stats, indent=2)}"
            await self._reply_text_or_document(update, payload, f"{station_name}_stats.json", caption=f"{station_name} stats")
        else:
            stats = {}
            for controller in self.radioListener.controllers.values():
                stats[controller.RADIO_CONF.get('NAME','UNKNOWN')] = controller.get_stats()
            payload = json.dumps(stats, indent=2)
            await self._reply_text_or_document(update, payload, "all_stats.json", caption="All stations stats")

    def send_message(self, text):
        if self.app is None:
            return
        chat_id = self.radioListener.CONFIG.get("TELEGRAM_CHAT_ID")
        if not chat_id:
            logger.log_event("TELEGRAM", "send_message skipped: TELEGRAM_CHAT_ID not set")
            return
        self._schedule(self._send_message(chat_id, text), f"send_message chat_id={chat_id}")

    def send_debug_message(self, text):
        if text:
            self.debug_history.append(text)
        if self.app is None:
            return
        if not self.radioListener.CONFIG.get("TELEGRAM_DEBUG_PRIVATE", False):
            return
        chat_id = self.radioListener.CONFIG.get("TELEGRAM_CHAT_ID_PRIVATE")
        if not chat_id:
            return
        self._schedule(self._send_message(chat_id, text), f"send_debug_message chat_id={chat_id}")

    def send_audio(self, audio_path, caption=""):
        if self.app is None:
            return
        chat_id = self.radioListener.CONFIG.get("TELEGRAM_CHAT_ID")
        if not chat_id:
            logger.log_event("TELEGRAM", "send_audio skipped: TELEGRAM_CHAT_ID not set")
            return
        self._schedule(self._send_audio(chat_id, audio_path, caption=caption), f"send_audio {audio_path}")
    
    def send_sms_message(self, phone_number, text = ""):
        if not text:
            text = "codeword"
        text = text.replace(" ", "%20")
        sms_url = f"sms:{phone_number}&body={text}"
        keyboard = [ [InlineKeyboardButton(f"SMS {text} to {phone_number}", callback_data=sms_url)] ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        #msg=f"sms:{phone_number}?body={text}"
        chat_id = self.radioListener.CONFIG.get("TELEGRAM_CHAT_ID")
        if not chat_id:
            logger.log_event("TELEGRAM", "send_sms_message skipped: TELEGRAM_CHAT_ID not set")
            return
        self._schedule(
            self._send_message(chat_id, "Send SMS?", reply_markup=reply_markup),
            f"send_sms_message chat_id={chat_id}",
        )

    @staticmethod
    async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Parses the CallbackQuery and updates the message text."""
        query = update.callback_query

        # CallbackQueries need to be answered, even if no notification to the user is needed
        # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
        await query.answer()
        
        await query.edit_message_text(text=f"{query.data}")
