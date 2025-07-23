You are an expert Telegram bot developer.

I have a bot file named `botv4.py`, and I want you to fully integrate two types of local database systems for managing user data:

---

📁 PART 1: JSON-Based Local Database System

🔹 Create a `database.json` file (if not exists) to store user data in dictionary format (user_id → data).
🔹 Add helper functions: `load_db()` and `save_db()` using `json.load()` and `json.dump()`
🔹 Add two admin-only features:
    - `/export_users`: Sends the full `database.json` file to the admin via Telegram.
    - When admin uploads a `.json` file to the bot, it auto-imports and replaces the current `database.json` and loads the data into memory immediately (no restart needed).

✅ This JSON system must:
- Be human-readable using `indent=2`
- Be auto-loaded on startup
- Be safe, error-handled, and only accessible to `admin_id = YOUR_ADMIN_ID_HERE`

---

🗃️ PART 2: SQLite Local Database Integration (using `import sqlite3`)

Also add full integration of an **SQLite-based database system** (`database.db`) with the following:

🔹 Create an SQLite table: `users (user_id INTEGER PRIMARY KEY, name TEXT, language TEXT, banned INTEGER DEFAULT 0)`
🔹 Add helper functions to:
    - Insert or update user data
    - Fetch user info by ID
    - List all users (for export)
🔹 Ensure all insert/update/read operations in the bot are backed by this SQLite database.
🔹 Connection should use `sqlite3.connect("database.db")` and be reusable.

---

✅ General Requirements:
- Use JSON for easy import/export
- Use SQLite for scalable backend storage
- Admin can trigger export/import of `.json` files
- User actions (e.g., start, command usage) should update both JSON and SQLite (or sync one-way if needed)
- Clean, optimized code
- Don't break any existing features in `botv4.py`

⚠️ Replace `5524867269` with my Telegram ID.

Please fully integrate all this into `botv4.py` and ensure it works locally with Termux or Python.

send me full working code and full bugs fix and no any error and make admin pannel bugs free and full working and send me full complete code