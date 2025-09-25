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

    # Handle reset/exit
    lowered = user_message.lower()
    if any(k in lowered for k in RESET_KEYWORDS):
        sessions[chat_id] = {"active": True, "history": []}
        await update.message.reply_text("🔄 Session reset. Start again with your symptoms.")
        return
    if any(k in lowered for k in EXIT_KEYWORDS):
        sessions[chat_id]["active"] = False
        await update.message.reply_text("✅ Triage ended. Send /start or 'reset' to begin again.")
        return

    # If triage inactive, ignore until restarted
    if not sessions[chat_id]["active"]:
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
