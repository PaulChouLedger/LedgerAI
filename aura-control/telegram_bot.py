# === telegram_bot.py — Aura Telegram Bridge with Multi-User Sessions ===
import os
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# === Load token ===
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AURA_CHAT_URL = os.getenv("AURA_CHAT_URL", "http://127.0.0.1:11434/chat")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("⚠️ TELEGRAM_BOT_TOKEN not found in environment")

# === Session state ===
sessions = {}  # { chat_id: {"active": bool, "history": []} }
RESET_KEYWORDS = {"reset", "restart", "new session"}
EXIT_KEYWORDS = {"exit", "stop", "end", "quit"}

# === Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    sessions[chat_id] = {"active": True, "history": []}
    await update.message.reply_text(
        "👋 Hello, I’m Aura via Telegram.\n"
        "You are now in triage mode. Type your symptoms to begin.\n"
        "Send 'reset' anytime to restart, or 'exit' to end."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.strip()
    chat_id = update.message.chat_id
    print(f"[Telegram] 📨 From {chat_id}: {user_message}")

    # Ensure user session exists
    if chat_id not in sessions:
        sessions[chat_id] = {"active": True, "history": []}
    
    print(f"[Telegram] 🔍 Session state for {chat_id}: {sessions[chat_id]}")

    # Handle reset/exit
    lowered = user_message.lower()
    if any(k in lowered for k in RESET_KEYWORDS):
        sessions[chat_id] = {"active": True, "history": []}
        print(f"[Telegram] 🔄 Reset: Session state for {chat_id}: {sessions[chat_id]}")
        # Forward reset to LLM container
        try:
            resp = requests.post(
                AURA_CHAT_URL,
                json={
                    "prompt": user_message,
                    "chat_id": str(chat_id),
                    "reset": True
                },
                timeout=10
            )
            if resp.status_code == 200:
                # Stream the reset response
                reply_buffer = ""
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if "<sentence_start>" in line:
                        reply_buffer = ""
                    elif "<sentence_end>" in line:
                        if reply_buffer.strip():
                            await update.message.reply_text(reply_buffer.strip())
                        reply_buffer = ""
                    else:
                        reply_buffer += line + " "
            else:
                await update.message.reply_text("🔄 Session reset. Start again with your symptoms.")
        except Exception as e:
            print(f"[Telegram] ❌ Reset error: {e}")
            await update.message.reply_text("🔄 Session reset. Start again with your symptoms.")
        return
    # Check for exit keywords as whole words only
    import re
    exit_pattern = r'\b(' + '|'.join(EXIT_KEYWORDS) + r')\b'
    if re.search(exit_pattern, lowered):
        matched_exits = re.findall(exit_pattern, lowered)
        print(f"[Telegram] 🚪 Exit keyword detected: {matched_exits}")
        sessions[chat_id]["active"] = False
        await update.message.reply_text("✅ Triage ended. Send /start or 'reset' to begin again.")
        return

    # If triage inactive, check if user is trying to start a new triage session
    if not sessions[chat_id]["active"]:
        print(f"[Telegram] ⚠️ Session inactive for {chat_id}, checking for medical symptoms...")
        # Check if the message contains medical symptoms that should trigger triage
        medical_keywords = ["pain", "ache", "headache", "chest", "abdominal", "stomach", "nausea", "dizzy", "fever", "cough", "shortness", "breath", "weakness", "numbness", "vision", "hearing", "speech", "difficulty"]
        if any(keyword in user_message.lower() for keyword in medical_keywords):
            # Reactivate session for medical symptoms
            sessions[chat_id]["active"] = True
            sessions[chat_id]["history"] = []
            print(f"[Telegram] 🔄 Reactivated session for medical symptoms: {user_message}")
            print(f"[Telegram] 🔄 Reactivated session state: {sessions[chat_id]}")
        else:
            print(f"[Telegram] ❌ No medical symptoms detected, session remains inactive")
            await update.message.reply_text("ℹ️ Triage not active. Send /start or 'reset' to begin.")
            return

    # Store message in history (optional, can pass to Aura)
    sessions[chat_id]["history"].append(user_message)

    try:
        # Forward to Aura’s /chat endpoint
        resp = requests.post(
            AURA_CHAT_URL,
            json={
                "prompt": user_message,
                "chat_id": str(chat_id),  # so backend can separate sessions if needed
                "history": sessions[chat_id]["history"]
            },
            stream=True,
            timeout=60
        )

        if resp.status_code != 200:
            await update.message.reply_text("⚠️ Error talking to Aura.")
            return

        # Collect streamed sentences
        reply_buffer = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if "<sentence_start>" in line:
                reply_buffer = ""
            elif "<sentence_end>" in line:
                if reply_buffer.strip():
                    await update.message.reply_text(reply_buffer.strip())
                reply_buffer = ""
            else:
                reply_buffer += line + " "

    except Exception as e:
        print(f"[Telegram] ❌ Error: {e}")
        await update.message.reply_text("⚠️ Could not connect to Aura.")

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", start))  # shortcut
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Telegram bot running... connected to Aura at", AURA_CHAT_URL)
    app.run_polling()

if __name__ == "__main__":
    main()
