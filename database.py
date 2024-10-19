import sqlite3

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('users.db')
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                password TEXT,
                face_image BLOB,
                keystroke_data TEXT
            )
        ''')
        self.conn.commit()
    
    def get_user(self, username):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'username': row[1],
                'password': row[2],
                'face_image': row[3],
                'keystroke_data': json.loads(row[4])
            }
        return None
    
    # ... other database methods ...
