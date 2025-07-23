#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import time
import psutil
import platform
import logging
import asyncio
import datetime
from typing import Dict, Optional, List

import aiohttp
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand
)
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- Configuration ---
TOKEN = '7965165313:AAHERd42_bVQY0YNZ4P8Ed2TJwqHs0ytASY'
OWNER_ID = 5524867269
ADMIN_ID = 5524867269
BOT_VERSION = "v2.1"  # Updated version
YOUR_USERNAME = '@patelmilan07'
UPDATE_CHANNEL = 'https://t.me/+w5gpsnKUVLVhYjhl'

# A4F API Configuration
A4F_API_URL = "https://api.a4f.co/v1/chat/completions"
A4F_IMAGE_API_URL = "https://api.a4f.co/v1/images/generations"
A4F_API_KEY = "ddc-a4f-9d06c9a8b0ad4098959c676b16336dac"
API_TIMEOUT = 30  # seconds
IMAGE_SIZE = "1024x1024"

# --- Model List ---
MODELS = {
    "💬 ChatGPT 4.5": "provider-6/gpt-4.1-nano",
    "🔍 DeepSeek R1": "provider-1/deepseek-r1-0528",
    "🧠 Meta llama 4": "provider-6/llama-4-scout",
    "🚀 Gemini Pro": "provider-6/gemini-2.5-flash",
    "🎨 Image Generator": "provider-4/imagen-4"  # Added image generation model
}
DEFAULT_MODEL = "provider-6/gpt-4.1-nano"

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Custom JSON Encoder ---
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)

# --- Local JSON Database Setup ---
class JSONDatabase:
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize collections
        self.users = self._load_collection("users")
        self.logs = self._load_collection("logs")
        self.banned_users = self._load_collection("banned_users")
        self.settings = self._load_collection("settings")
        self.image_logs = self._load_collection("image_logs")
        
        # Initialize default settings
        if "maintenance" not in self.settings:
            self.settings["maintenance"] = {"_id": "maintenance", "active": False}
            self._save_collection("settings", self.settings)
    
    def _load_collection(self, name: str) -> Dict:
        path = os.path.join(self.data_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}
    
    def _save_collection(self, name: str, data: Dict) -> None:
        path = os.path.join(self.data_dir, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=DateTimeEncoder)
    
    def find_one(self, collection: str, query: Dict) -> Optional[Dict]:
        coll = getattr(self, collection)
        for key, item in coll.items():
            match = True
            for k, v in query.items():
                if item.get(k) != v:
                    match = False
                    break
            if match:
                return item
        return None
    
    def insert_one(self, collection: str, data: Dict) -> None:
        coll = getattr(self, collection)
        if "_id" not in data:
            data["_id"] = str(len(coll) + 1)
        coll[data["_id"]] = data
        self._save_collection(collection, coll)
    
    def update_one(self, collection: str, query: Dict, update: Dict) -> None:
        coll = getattr(self, collection)
        item = self.find_one(collection, query)
        if item:
            if "$set" in update:
                item.update(update["$set"])
            if "$inc" in update:
                for k, v in update["$inc"].items():
                    item[k] = item.get(k, 0) + v
            if "$setOnInsert" in update:
                for k, v in update["$setOnInsert"].items():
                    if k not in item:
                        item[k] = v
            coll[item["_id"]] = item
            self._save_collection(collection, coll)
    
    def count_documents(self, collection: str, query: Dict = None) -> int:
        coll = getattr(self, collection)
        if not query:
            return len(coll)
        count = 0
        for item in coll.values():
            match = True
            for k, v in query.items():
                if item.get(k) != v:
                    match = False
                    break
            if match:
                count += 1
        return count
    
    def delete_one(self, collection: str, query: Dict) -> int:
        coll = getattr(self, collection)
        item = self.find_one(collection, query)
        if item:
            del coll[item["_id"]]
            self._save_collection(collection, coll)
            return 1
        return 0
    
    def aggregate(self, collection: str, pipeline: List[Dict]) -> List[Dict]:
        # Simplified aggregation for basic stats
        coll = getattr(self, collection)
        results = []
        
        for stage in pipeline:
            if "$group" in stage:
                group_by = stage["$group"]["_id"]
                results = []
                groups = {}
                
                for item in coll.values():
                    if isinstance(group_by, str):
                        key = item.get(group_by)
                    else:
                        key = tuple(item.get(field) for field in group_by)
                    
                    if key not in groups:
                        groups[key] = {"_id": key, "count": 0}
                    
                    if "$sum" in stage["$group"]:
                        groups[key]["count"] += 1
                
                results = list(groups.values())
            
            if "$sort" in stage:
                field = list(stage["$sort"].keys())[0]
                reverse = stage["$sort"][field] == -1
                results.sort(key=lambda x: x.get(field, 0), reverse=reverse)
        
        return results

# Initialize database
db = JSONDatabase()

# --- Helper Functions ---
def get_maintenance_status() -> bool:
    """Get current maintenance status from DB"""
    status = db.find_one("settings", {"_id": "maintenance"})
    return status.get("active", False) if status else False

def set_maintenance_status(status: bool) -> None:
    """Update maintenance status in DB"""
    db.update_one(
        "settings",
        {"_id": "maintenance"},
        {"$set": {"active": status}}
    )

async def check_user(user_id: int) -> Dict:
    """Ensure user exists in DB, return user data"""
    user_data = db.find_one("users", {"user_id": user_id})
    if not user_data:
        new_user = {
            "_id": str(user_id),
            "user_id": user_id,
            "model": DEFAULT_MODEL,
            "joined": datetime.datetime.utcnow().isoformat(),
            "first_start": True,
            "requests": 0,
            "image_requests": 0,
            "last_active": datetime.datetime.utcnow().isoformat()
        }
        db.insert_one("users", new_user)
        return new_user
    
    # Convert string dates back to datetime if needed
    if isinstance(user_data.get("joined"), str):
        user_data["joined"] = datetime.datetime.fromisoformat(user_data["joined"])
    if isinstance(user_data.get("last_active"), str):
        user_data["last_active"] = datetime.datetime.fromisoformat(user_data["last_active"])
    
    return user_data

async def is_banned(user_id: int) -> bool:
    """Check if user is banned"""
    return db.find_one("banned_users", {"user_id": user_id}) is not None

async def log_request(user_id: int, model: str, prompt: str) -> None:
    """Log user request to database"""
    log_entry = {
        "_id": f"{user_id}-{time.time()}",
        "user_id": user_id,
        "model": model,
        "prompt": prompt[:200],  # Truncate long prompts
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "type": "image" if model == "provider-4/imagen-4" else "text"
    }
    
    if model == "provider-4/imagen-4":
        db.insert_one("image_logs", log_entry)
        # Update user's image request count
        db.update_one(
            "users",
            {"user_id": user_id},
            {"$inc": {"image_requests": 1}}
        )
    else:
        db.insert_one("logs", log_entry)
    
    # Update user's last active time and increment request count
    db.update_one(
        "users",
        {"user_id": user_id},
        {"$set": {"last_active": datetime.datetime.utcnow().isoformat()},
         "$inc": {"requests": 1}}
    )

def backup_db() -> str:
    """Create database backup to JSON file"""
    try:
        backup_dir = "backups"
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"backup_{timestamp}.json")
        
        backup_data = {
            "users": db.users,
            "logs": db.logs,
            "image_logs": db.image_logs,
            "banned_users": db.banned_users,
            "settings": db.settings
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, cls=DateTimeEncoder)
        
        logger.info(f"Backup created at {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return ""

# --- API Caller Functions ---
async def generate_image(prompt: str) -> str:
    """Generate image using A4F API"""
    headers = {
        "Authorization": f"Bearer {A4F_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "provider-4/imagen-4",
        "prompt": prompt,
        "n": 1,
        "size": IMAGE_SIZE
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                A4F_IMAGE_API_URL,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Image API error {resp.status}: {error}")
                    return f"❌ Image Generation Failed (Status: {resp.status})"

                res = await resp.json()
                if 'data' in res and len(res['data']) > 0:
                    return res['data'][0]['url']  # Return the image URL
                return "❌ No image was generated"
    except asyncio.TimeoutError:
        logger.warning("Image generation API timeout")
        return "⏳ Image generation is taking too long. Please try again later."
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return f"❌ Image Generation Error: {str(e)}"

async def call_a4f_api(prompt: str, model: str) -> str:
    """Call A4F API with timeout handling"""
    headers = {
        "Authorization": f"Bearer {A4F_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Check if the model is for image generation
    if model == "provider-4/imagen-4":
        return await generate_image(prompt)
    
    # Regular text completion
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                A4F_API_URL,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"API error {resp.status}: {error}")
                    return f"❌ API Error (Status: {resp.status})"

                res = await resp.json()
                return res['choices'][0]['message']['content']
    except asyncio.TimeoutError:
        logger.warning(f"API timeout for model {model}")
        return "⏳ The API is taking too long to respond. Please try again later."
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return f"❌ API Error: {str(e)}"

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command with welcome message (only once)"""
    if get_maintenance_status() and update.effective_user.id not in [OWNER_ID, ADMIN_ID]:
        await update.message.reply_text("🔒 Bot is currently under maintenance. Please try again later.")
        return

    user_id = update.effective_user.id
    
    if await is_banned(user_id):
        await update.message.reply_text("🚫 You are banned from using this bot.")
        return

    user_data = await check_user(user_id)
    
    if user_data.get('first_start', True):
        welcome_msg = (
            f"👋 Welcome to *Multi-AI Bot* {update.effective_user.first_name}!\n\n"
            "✨ I support multiple AI models:\n"
            "• 💬 ChatGPT 4.5 (Fastest)\n"
            "• 🔍 DeepSeek R1 (Best for research)\n"
            "• 🧠 Meta llama 4 (Most creative)\n"
            "• 🚀 Gemini Pro (Best for coding)\n"
            "• 🎨 Image Generator (AI Art)\n\n"
            "🔹 Use /model to choose your AI model\n"
            "🔹 Just type your message to chat\n"
            "🔹 Use /help for all commands\n\n"
            f"📢 Updates: {UPDATE_CHANNEL}"
        )
        db.update_one(
            "users",
            {"user_id": user_id},
            {"$set": {"first_start": False}}
        )
    else:
        welcome_msg = "You're already registered! Just send me a message."

    await update.message.reply_text(welcome_msg, parse_mode=ParseMode.MARKDOWN)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command"""
    help_text = (
        "📚 *Available Commands:*\n\n"
        "👤 User Commands:\n"
        "/start - Initialize the bot\n"
        "/model - Change AI model\n"
        "/mymodel - Show current model\n"
        "/resetmodel - Reset to default model\n"
        "/ping - Check bot status\n"
        "/menu - Show interactive menu\n"
        "/help - This message\n\n"
        "🛠 Admin Commands:\n"
        "/stats - Show bot statistics\n"
        "/lockbot - Enable maintenance mode\n"
        "/unlockbot - Disable maintenance mode\n"
        "/banuser [id] - Ban a user\n"
        "/unbanuser [id] - Unban a user\n"
        "/broadcast [msg] - Broadcast message\n"
        f"\n👨‍💻 Admin: {YOUR_USERNAME}"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show interactive menu with inline buttons"""
    keyboard = [
        [InlineKeyboardButton("💬 Change Model", callback_data="model_menu")],
        [InlineKeyboardButton("🛠 My Settings", callback_data="my_settings")],
        [InlineKeyboardButton("📊 Bot Stats", callback_data="bot_stats")],
        [InlineKeyboardButton("🆘 Help", callback_data="help_menu")]
    ]
    if update.effective_user.id in [OWNER_ID, ADMIN_ID]:
        keyboard.append([InlineKeyboardButton("⚙️ Admin Panel", callback_data="admin_panel")])
    
    await update.message.reply_text(
        "📱 *Main Menu* - Choose an option:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model command with inline keyboard"""
    keyboard = [
        [InlineKeyboardButton(name, callback_data=f"model_{code}")]
        for name, code in MODELS.items()
    ]
    await update.message.reply_text(
        "🤖 *Select your preferred AI model:*",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

async def mymodel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /mymodel command"""
    user_id = update.effective_user.id
    user_data = await check_user(user_id)
    model_name = next((k for k, v in MODELS.items() if v == user_data['model']), user_data['model'])
    await update.message.reply_text(f"🛠 Your current model: *{model_name}*", parse_mode=ParseMode.MARKDOWN)

async def resetmodel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /resetmodel command"""
    user_id = update.effective_user.id
    db.update_one(
        "users",
        {"user_id": user_id},
        {"$set": {"model": DEFAULT_MODEL}}
    )
    model_name = next((k for k, v in MODELS.items() if v == DEFAULT_MODEL), DEFAULT_MODEL)
    await update.message.reply_text(f"🔄 Model reset to default: *{model_name}*", parse_mode=ParseMode.MARKDOWN)

async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ping command with system stats"""
    # Calculate latency
    start_time = time.time()
    message = await update.message.reply_text("🏓 Pinging...")
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # ms
    
    # Get system info
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    os_name = platform.system()
    bot_uptime = str(datetime.timedelta(seconds=time.time() - psutil.Process(os.getpid()).create_time()))
    
    ping_msg = (
        f"🏓 *Pong!*\n\n"
        f"⚡ Response Time: `{latency:.2f} ms`\n"
        f"🤖 Bot Version: `{BOT_VERSION}`\n"
        f"⏱ Uptime: `{bot_uptime}`\n\n"
        f"🖥 *System Stats:*\n"
        f"• OS: `{os_name}`\n"
        f"• CPU: `{cpu_percent}%`\n"
        f"• RAM: `{memory.percent}%`\n"
        f"• Disk: `{disk.percent}%`\n"
        f"• Host: `Termux (Local)`"
    )
    
    await message.edit_text(ping_msg, parse_mode=ParseMode.MARKDOWN)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command with detailed statistics"""
    if update.effective_user.id not in [OWNER_ID, ADMIN_ID]:
        await update.message.reply_text("🚫 Admin only command")
        return
        
    try:
        # Basic stats
        total_users = db.count_documents("users")
        total_requests = db.count_documents("logs")
        total_images = db.count_documents("image_logs")
        banned_count = db.count_documents("banned_users")
        
        # Model-wise stats
        model_stats = db.aggregate("logs", [
            {"$group": {"_id": "$model", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ])
        
        # Top users
        top_users = sorted(
            db.users.values(),
            key=lambda x: x.get("requests", 0),
            reverse=True
        )[:5]
        
        # Format stats
        model_text = "\n".join(
            f"• {next((k for k, v in MODELS.items() if v == stat['_id']), stat['_id'])}: {stat['count']}"
            for stat in model_stats
        )
        
        user_text = "\n".join(
            f"• {user.get('first_name', 'User')} ({user['user_id']}): {user.get('requests', 0)} requests ({user.get('image_requests', 0)} images)"
            for user in top_users
        )
        
        stats_msg = (
            f"📊 *Bot Statistics*\n\n"
            f"👤 Total Users: `{total_users}`\n"
            f"📨 Total Requests: `{total_requests}`\n"
            f"🖼 Total Images: `{total_images}`\n"
            f"🚫 Banned Users: `{banned_count}`\n"
            f"🔒 Maintenance: `{'ON' if get_maintenance_status() else 'OFF'}`\n\n"
            f"🤖 *Model Usage*\n{model_text}\n\n"
            f"🏆 *Top Users*\n{user_text}"
        )
        
        await update.message.reply_text(stats_msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        await update.message.reply_text("❌ Failed to fetch statistics")

async def lock_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enable maintenance mode"""
    if update.effective_user.id not in [OWNER_ID, ADMIN_ID]:
        await update.message.reply_text("🚫 Admin only command")
        return
        
    set_maintenance_status(True)
    await update.message.reply_text("🔒 *Bot has been locked for maintenance*", parse_mode=ParseMode.MARKDOWN)

async def unlock_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Disable maintenance mode"""
    if update.effective_user.id not in [OWNER_ID, ADMIN_ID]:
        await update.message.reply_text("🚫 Admin only command")
        return
        
    set_maintenance_status(False)
    await update.message.reply_text("🔓 *Bot has been unlocked*", parse_mode=ParseMode.MARKDOWN)

async def ban_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin: Ban a user"""
    if update.effective_user.id not in [OWNER_ID, ADMIN_ID]:
        await update.message.reply_text("🚫 Admin only command")
        return
    
    try:
        user_id = int(context.args[0])
        # Check if already banned
        if db.find_one("banned_users", {"user_id": user_id}):
            await update.message.reply_text("ℹ️ User is already banned")
            return
            
        db.insert_one("banned_users", {
            "_id": str(user_id),
            "user_id": user_id,
            "banned_by": update.effective_user.id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "reason": " ".join(context.args[1:]) if len(context.args) > 1 else "No reason provided"
        })
        await update.message.reply_text(f"✅ User `{user_id}` banned", parse_mode=ParseMode.MARKDOWN)
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /banuser <user_id> [reason]")
    except Exception as e:
        logger.error(f"Ban error: {e}")
        await update.message.reply_text("❌ Failed to ban user")

async def unban_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin: Unban a user"""
    if update.effective_user.id not in [OWNER_ID, ADMIN_ID]:
        await update.message.reply_text("🚫 Admin only command")
        return
    
    try:
        user_id = int(context.args[0])
        result = db.delete_one("banned_users", {"user_id": user_id})
        if result > 0:
            await update.message.reply_text(f"✅ User `{user_id}` unbanned", parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text("ℹ️ User was not banned")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /unbanuser <user_id>")
    except Exception as e:
        logger.error(f"Unban error: {e}")
        await update.message.reply_text("❌ Failed to unban user")

async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin: Broadcast message to all users"""
    if update.effective_user.id not in [OWNER_ID, ADMIN_ID]:
        await update.message.reply_text("🚫 Admin only command")
        return
        
    if not context.args:
        await update.message.reply_text("Usage: /broadcast <message>")
        return
        
    message = " ".join(context.args)
    all_users = db.users.values()
    total = 0
    success = 0
    
    status_msg = await update.message.reply_text("📢 Starting broadcast...")
    
    for user in all_users:
        try:
            await context.bot.send_message(
                chat_id=user["user_id"],
                text=f"📢 Announcement from admin:\n\n{message}"
            )
            success += 1
        except Exception as e:
            logger.error(f"Broadcast failed to {user['user_id']}: {e}")
        total += 1
        
        # Update status every 10 messages
        if total % 10 == 0:
            await status_msg.edit_text(
                f"📢 Broadcasting...\nSent: {success}/{total}",
                parse_mode=ParseMode.MARKDOWN
            )
        
        # Avoid rate limits
        await asyncio.sleep(0.1)
    
    await status_msg.edit_text(
        f"✅ Broadcast completed!\n• Sent: {success}\n• Failed: {total - success}",
        parse_mode=ParseMode.MARKDOWN
    )

# --- Callback Handlers ---
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all callback queries"""
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == "model_menu":
        await model_command(update, context)
    elif data.startswith("model_"):
        await model_callback(update, context)
    elif data == "my_settings":
        await mymodel_command(update, context)
    elif data == "bot_stats":
        await stats_command(update, context)
    elif data == "help_menu":
        await help_command(update, context)
    elif data == "admin_panel":
        if query.from_user.id in [OWNER_ID, ADMIN_ID]:
            keyboard = [
                [InlineKeyboardButton("📊 Stats", callback_data="bot_stats")],
                [InlineKeyboardButton("🔒 Lock Bot", callback_data="lock_bot")],
                [InlineKeyboardButton("🔓 Unlock Bot", callback_data="unlock_bot")],
                [InlineKeyboardButton("📢 Broadcast", callback_data="broadcast_menu")],
                [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
            ]
            await query.edit_message_text(
                "⚙️ *Admin Panel*",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await query.edit_message_text("🚫 Admin only")
    elif data == "lock_bot":
        set_maintenance_status(True)
        await query.edit_message_text("🔒 *Bot locked for maintenance*", parse_mode=ParseMode.MARKDOWN)
    elif data == "unlock_bot":
        set_maintenance_status(False)
        await query.edit_message_text("🔓 *Bot unlocked*", parse_mode=ParseMode.MARKDOWN)
    elif data == "broadcast_menu":
        await query.edit_message_text("📢 Enter broadcast message using /broadcast command")
    elif data == "main_menu":
        await menu_command(update, context)

async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle model selection from inline keyboard"""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    model_code = query.data.replace("model_", "")
    model_name = next((k for k, v in MODELS.items() if v == model_code), model_code)

    db.update_one(
        "users",
        {"user_id": user_id},
        {"$set": {"model": model_code}}
    )
    await query.edit_message_text(
        f"✅ Model changed to: *{model_name}*\n\n"
        f"Now you can start chatting with your new AI model!",
        parse_mode=ParseMode.MARKDOWN
    )

# --- Message Handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all text messages"""
    if get_maintenance_status() and update.effective_user.id not in [OWNER_ID, ADMIN_ID]:
        await update.message.reply_text("🔒 Bot is currently under maintenance. Please try again later.")
        return

    user_id = update.effective_user.id
    
    if await is_banned(user_id):
        await update.message.reply_text("🚫 You are banned from using this bot.")
        return

    prompt = update.message.text.strip()
    if not prompt:
        await update.message.reply_text("Please send a valid message.")
        return

    user_data = await check_user(user_id)
    model = user_data.get("model", DEFAULT_MODEL)

    # Log request
    await log_request(user_id, model, prompt)

    # Show appropriate chat action
    action = ChatAction.UPLOAD_PHOTO if model == "provider-4/imagen-4" else ChatAction.TYPING
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=action
    )

    # Call API and respond
    reply = await call_a4f_api(prompt, model)
    
    # Handle image response
    if model == "provider-4/imagen-4":
        if reply.startswith(("http://", "https://")):
            await update.message.reply_photo(
                photo=reply,
                caption=f"🖼 Generated image for: '{prompt}'"
            )
        else:
            await update.message.reply_text(reply)
    else:
        await update.message.reply_text(
            reply,
            parse_mode=ParseMode.MARKDOWN
        )

# --- Error Handler ---
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors and send notification to admin"""
    logger.error(f"Update {update} caused error: {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ An error occurred. Please try again later."
        )
    
    # Notify admin
    try:
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=f"⚠️ Bot Error:\n{context.error}\n\nUpdate: {update.to_dict() if update else 'None'}"
        )
    except Exception as e:
        logger.error(f"Failed to send error notification: {e}")

# --- Main Function ---
def main() -> None:
    """Start the bot with all handlers"""
    # Initialize application
    app = ApplicationBuilder().token(TOKEN).build()
    
    # Register commands for Telegram menu
    commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("help", "Show help information"),
        BotCommand("model", "Change AI model"),
        BotCommand("mymodel", "Show current model"),
        BotCommand("resetmodel", "Reset to default model"),
        BotCommand("menu", "Show interactive menu"),
        BotCommand("ping", "Check bot status"),
        BotCommand("stats", "Show bot statistics (Admin)"),
    ]
    app.bot.set_my_commands(commands)

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("mymodel", mymodel_command))
    app.add_handler(CommandHandler("resetmodel", resetmodel_command))
    app.add_handler(CommandHandler("ping", ping_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("lockbot", lock_bot))
    app.add_handler(CommandHandler("unlockbot", unlock_bot))
    app.add_handler(CommandHandler("banuser", ban_user))
    app.add_handler(CommandHandler("unbanuser", unban_user))
    app.add_handler(CommandHandler("broadcast", broadcast))

    # Callback handler
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Message handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("🤖 Bot is starting...")
    logger.info(f"Bot version: {BOT_VERSION}")
    logger.info(f"Models available: {', '.join(MODELS.keys())}")
    logger.info("Using local JSON database storage")
    
    # Start the bot
    app.run_polling()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Bot crashed: {e}")
        raise