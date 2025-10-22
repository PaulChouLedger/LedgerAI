# === telegram_bot.py — Aura Telegram Bridge with Multi-User Sessions ===
import os
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# === Load token from workspace root .env ===
# Load .env from workspace root (2 levels up from this file)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(workspace_root, '.env')
load_dotenv(dotenv_path)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AURA_CHAT_URL = os.getenv("AURA_CHAT_URL", "http://127.0.0.1:11434/chat-tg")
# Debug info is now always shown in terminal logs (not in Telegram messages)

if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token":
    raise RuntimeError(
        "❌ Missing Telegram bot token!\n"
        "   Run: ./aura_config.sh\n"
        "   Choose option 6 to configure Telegram bot\n"
        "   Or edit .env and set: TELEGRAM_BOT_TOKEN=your_token_here\n"
        "   Get a token from @BotFather on Telegram"
    )

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
        # Clear both Telegram session and LLM container state
        sessions[chat_id] = {"active": True, "history": []}
        print(f"[Telegram] 🔄 Reset: Session state for {chat_id}: {sessions[chat_id]}")
        # Forward reset to LLM container using streaming endpoint
        try:
            resp = requests.post(
                AURA_CHAT_URL,
                json={
                    "prompt": user_message,
                    "chat_id": str(chat_id),
                    "reset": True
                },
                timeout=10,
                stream=True
            )
            if resp.status_code == 200:
                # Collect all streaming responses (same logic as TTS)
                response_parts = []
                for line in resp.iter_lines(decode_unicode=True):
                    token = line.strip()
                    if not token:
                        continue
                    response_parts.append(token)
                
                response_text = ' '.join(response_parts) if response_parts else "Session reset. Start again with your symptoms."
                
                # Remove control tags for Telegram display
                import re
                response_text = re.sub(r'<sentence_start>|<sentence_end>|<pause>', '', response_text)
                response_text = re.sub(r'\s+', ' ', response_text).strip()  # Normalize spaces
                
                await update.message.reply_text(response_text)
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
    
    # If session is active, all messages should be processed (including greetings during triage)
    print(f"[Telegram] ✅ Session active for {chat_id}, processing message: {user_message}")

    # Store message in history (optional, can pass to Aura)
    sessions[chat_id]["history"].append(user_message)

    try:
        # Forward to Aura's /chat-tg endpoint (returns JSON, not streaming)
        resp = requests.post(
            AURA_CHAT_URL,
            json={
                "prompt": user_message,
                "chat_id": str(chat_id)  # Backend uses this for session management
            },
            timeout=30
        )

        if resp.status_code != 200:
            await update.message.reply_text("⚠️ Error talking to Aura.")
            return

        # Parse JSON response from /chat-tg
        try:
            import json
            response_data = resp.json()
            response_text = response_data.get("response", "I'm sorry, I didn't understand that.")
            debug_data = response_data.get("debug")  # Optional debug info from adaptive engine
        except json.JSONDecodeError:
            print(f"[Telegram] ❌ Failed to parse JSON: {resp.text}")
            response_text = "I'm sorry, there was an error processing your request."
            debug_data = None
        
        # Debug: Log the response to help diagnose issues
        print(f"[Telegram] 📤 Response to {chat_id}: {response_text[:100]}...")
        
        # Print debug info to terminal (always, for monitoring)
        if debug_data:
            print(f"\n{'='*80}")
            print(f"[Telegram] 🔍 INTERNAL REASONING (Session: {chat_id})")
            print(f"{'='*80}")
            
            # Matching algorithm info (if this is initial assessment)
            if 'matching' in debug_data:
                match_info = debug_data['matching']
                print(f"[Telegram] 🎯 MATCHING ALGORITHM:")
                print(f"[Telegram]    Mode: {match_info.get('mode', 'unknown')}")
                print(f"[Telegram]    Strategy: {match_info.get('strategy', 'unknown')}")
                if 'thresholds' in match_info:
                    thresh = match_info['thresholds']
                    print(f"[Telegram]    Thresholds:")
                    print(f"[Telegram]       - Char overlap: >{thresh.get('char_overlap', 0.75)}")
                    print(f"[Telegram]       - Semantic: >{thresh.get('semantic', 0.88)}")
                if 'matched_count' in match_info:
                    print(f"[Telegram]    Matched: {match_info['matched_count']} guidelines")
                if 'filtered_count' in match_info:
                    print(f"[Telegram]    Filtered: {match_info['filtered_count']} guidelines")
                if 'timing' in match_info:
                    print(f"[Telegram]    Time: {match_info['timing']:.2f}s")
                print(f"[Telegram]    ---")
            
            if 'demographics' in debug_data:
                demo = debug_data['demographics']
                print(f"[Telegram] 👤 Patient: {demo.get('age', '?')} y/o {demo.get('sex', '?')}")
            
            if 'question_number' in debug_data:
                print(f"[Telegram] 📝 Question: #{debug_data['question_number']}")
            
            if 'oldcarts_coverage' in debug_data:
                print(f"[Telegram] 📋 OLDCARTS: {debug_data['oldcarts_coverage']} ({debug_data.get('oldcarts_count', '?/8')})")
            
            if debug_data.get('clarification_counts'):
                clarif = debug_data['clarification_counts']
                if clarif:
                    clarif_str = ', '.join([f"{k}:{v}" for k, v in clarif.items()])
                    print(f"[Telegram] 🔁 Clarifications: {clarif_str}")
            
            if debug_data.get('last_answer'):
                print(f"[Telegram] 💬 Last Answer: '{debug_data['last_answer']}'")
            
            if 'pool_status' in debug_data:
                pool = debug_data['pool_status']
                print(f"[Telegram] 📊 Pool: Active={pool['active']}, Reserve={pool['reserve']}, Ruled out={pool['ruled_out']}")
            
            if 'active_differentials' in debug_data and debug_data['active_differentials']:
                print(f"[Telegram] 📊 TOP DIFFERENTIALS:")
                for diff in debug_data['active_differentials']:
                    urgency_emoji = "🚨" if diff['urgency'] == 'emergent' else "⚠️" if diff['urgency'] == 'urgent' else "📋"
                    print(f"[Telegram]   {diff['rank']}. {diff['name']} ({diff['score']}) {urgency_emoji}")
            
            if debug_data.get('last_answer_scores'):
                scores = debug_data['last_answer_scores']
                print(f"[Telegram] 🎯 Answer Scores: {scores}")
            
            # Enhanced debugging for failure cases
            if 'active_guidelines' in debug_data:
                print(f"[Telegram] 📊 Active Guidelines: {debug_data['active_guidelines']}")
            if 'reserve_pool' in debug_data:
                print(f"[Telegram] 📊 Reserve Pool: {debug_data['reserve_pool']}")
            if 'ruled_out' in debug_data:
                print(f"[Telegram] 📊 Ruled Out: {debug_data['ruled_out']}")
            if 'oldcarts_covered' in debug_data:
                print(f"[Telegram] 📋 OLDCARTS Coverage: {debug_data['oldcarts_covered']}")
            if 'demographics' in debug_data:
                print(f"[Telegram] 👤 Demographics: {debug_data['demographics']}")
            
            print(f"{'='*80}\n")
        
        # Remove control tags for Telegram display
        import re
        clean_response = re.sub(r'<sentence_start>|<sentence_end>|<pause>', '', response_text)
        clean_response = re.sub(r'\s+', ' ', clean_response).strip()  # Normalize spaces
        
        # Send the main response
        await update.message.reply_text(clean_response)
        
        # Debug info is now ONLY in terminal logs (not sent to Telegram)
        # This keeps the chat clean while still providing visibility for monitoring
        
        # Check if this looks like a triage completion
        triage_completed = any(phrase in response_text.lower() for phrase in [
            "classified as", "seek emergency", "call 911", "see a doctor", 
            "medical evaluation", "emergency care"
        ])
        
        # Clear session state after triage completion
        if triage_completed:
            sessions[chat_id] = {"active": True, "history": []}
            print(f"[Telegram] 🔄 Auto-reset after triage completion for {chat_id}: {sessions[chat_id]}")
            # Also send a reset command to LLM container to clear its state
            try:
                requests.post(
                    AURA_CHAT_URL,
                    json={
                        "prompt": "reset",
                        "chat_id": str(chat_id),
                        "reset": True
                    },
                    timeout=5,
                    stream=True
                )
                print(f"[Telegram] 🔄 Sent reset command to LLM container for {chat_id}")
            except Exception as e:
                print(f"[Telegram] ⚠️ Failed to reset LLM container: {e}")

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
