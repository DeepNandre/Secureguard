from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import cv2
import numpy as np
import base64
import json
import sqlite3
import face_recognition
import bcrypt
import pyotp
import qrcode
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepface import DeepFace
import speech_recognition as sr
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('users.db', check_same_thread=False)
        self.create_tables()
        self.ensure_keystroke_data_column()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                face_encoding TEXT,
                keystroke_data TEXT
            )
        ''')
        self.conn.commit()

    def ensure_keystroke_data_column(self):
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'keystroke_data' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN keystroke_data TEXT')
            self.conn.commit()

    def add_user(self, username, password, face_encoding_base64, keystroke_data):
        cursor = self.conn.cursor()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        keystroke_data_json = json.dumps(keystroke_data)
        try:
            cursor.execute('''
                INSERT INTO users (username, password_hash, face_encoding, keystroke_data)
                VALUES (?, ?, ?, ?)
            ''', (username, password_hash, face_encoding_base64, keystroke_data_json))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user(self, username):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        if user:
            return {
                'username': user[1],
                'password_hash': user[2],
                'face_encoding': user[3],
                'keystroke_data': user[4]  # This is already a JSON string
            }
        return None

    def get_all_users(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users')
        users = cursor.fetchall()
        return [{
            'id': user[0],
            'username': user[1],
            'password_hash': user[2],
            'face_encoding': user[3],
            'keystroke_data': json.loads(user[4])
        } for user in users]

    def get_password_hash(self, username):
        cursor = self.conn.cursor()
        cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        return result[0] if result else None

class KeystrokeDynamicsModel:
    def __init__(self):
        self.model = RandomForestClassifier()
    
    def preprocess_data(self, keystroke_data):
        features = []
        for data in keystroke_data:
            hold_times = [t['timestamp'] - data['timestamps'][0] for t in data['timestamps']]
            interval_times = [data['timestamps'][i+1] - data['timestamps'][i] for i in range(len(data['timestamps'])-1)]
            features.append(hold_times + interval_times)
        return np.array(features)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained. Accuracy: {accuracy}")

    def predict(self, X):
        return self.model.predict(X)

class SmartAuthenticationSystem:
    def __init__(self, db):
        self.db = db

    def register(self, username, password, face_image, keystroke_data):
        if self.db.get_user(username):
            return False, "Username already exists"
        
        # Process face image
        try:
            face_image_np = np.frombuffer(base64.b64decode(face_image.split(',')[1]), dtype=np.uint8)
            face_image_cv = cv2.imdecode(face_image_np, cv2.IMREAD_COLOR)
            
            face_locations = face_recognition.face_locations(face_image_cv)
            if not face_locations:
                return False, "No face detected in the image"
            
            face_encodings = face_recognition.face_encodings(face_image_cv, face_locations)
            if not face_encodings:
                return False, "Unable to encode face in the image"
            
            face_encoding = face_encodings[0]
            face_encoding_base64 = base64.b64encode(face_encoding.tobytes()).decode('utf-8')
            
            self.db.add_user(username, password, face_encoding_base64, keystroke_data)
            return True, "Registration successful"
        except IndexError:
            return False, "Error processing face image. Please try again with a clearer image."
        except Exception as e:
            app.logger.error(f"Registration error: {str(e)}")
            return False, f"Registration failed: {str(e)}"

    def login(self, username, password, face_image, keystroke_data):
        user = self.db.get_user(username)
        if not user:
            return False, "User not found"
        
        # Check password
        stored_password_hash = user['password_hash']
        if isinstance(stored_password_hash, str):
            stored_password_hash = stored_password_hash.encode('utf-8')
        
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password_hash):
            return False, "Incorrect password"
        
        # Face recognition
        try:
            stored_face_encoding = np.frombuffer(base64.b64decode(user['face_encoding']), dtype=np.float64)
            face_image_np = np.frombuffer(base64.b64decode(face_image.split(',')[1]), dtype=np.uint8)
            face_image_cv = cv2.imdecode(face_image_np, cv2.IMREAD_COLOR)
            
            app.logger.info(f"Received face image: {face_image}")
            app.logger.info(f"Decoded face image shape: {face_image_cv.shape}")
            
            face_locations = face_recognition.face_locations(face_image_cv)
            if not face_locations:
                return False, "No face detected in the login image"
            
            login_face_encoding = face_recognition.face_encodings(face_image_cv, face_locations)[0]
            face_distance = face_recognition.face_distance([stored_face_encoding], login_face_encoding)[0]
            
            if face_distance > 0.6:  # Adjust this threshold as needed
                return False, "Face does not match"
        except Exception as e:
            return False, f"Error during face recognition: {str(e)}"
        
        # Keystroke dynamics verification
        try:
            stored_keystroke_data = json.loads(user['keystroke_data']) if user['keystroke_data'] else None
            if stored_keystroke_data and keystroke_data:
                if not self.verify_keystroke_dynamics(stored_keystroke_data, keystroke_data):
                    return False, "Keystroke pattern does not match"
            else:
                app.logger.warning("Keystroke data missing for user or login attempt")
        except Exception as e:
            app.logger.error(f"Error during keystroke verification: {str(e)}")
            # Continue with login process even if keystroke verification fails
        
        return True, "Login successful"

    def verify_keystroke_dynamics(self, stored_data, new_data):
        # Extract timestamps from the stored and new data
        stored_timestamps = [event['timestamp'] for event in stored_data]
        new_timestamps = new_data['timestamps']
        
        if len(stored_timestamps) != len(new_timestamps):
            return False
        
        # Compare the intervals between keystrokes
        stored_intervals = [stored_timestamps[i+1] - stored_timestamps[i] for i in range(len(stored_timestamps)-1)]
        new_intervals = [new_timestamps[i+1] - new_timestamps[i] for i in range(len(new_timestamps)-1)]
        
        # Allow for some tolerance in the timing
        tolerance = 200  # milliseconds
        for s, n in zip(stored_intervals, new_intervals):
            if abs(s - n) > tolerance:
                return False
        
        return True

auth_system = SmartAuthenticationSystem(Database())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        username = request.form['username']
        password = request.form['password']
        face_image = request.form['face_image']
        keystroke_data = json.loads(request.form['keystroke_data'])

        app.logger.info(f"Received registration request for user: {username}")
        app.logger.info(f"Keystroke data received: {keystroke_data}")

        # Ensure keystroke_data is in the correct format
        formatted_keystroke_data = [
            {'key': event['key'], 'timestamp': event['timestamp']}
            for event in keystroke_data
        ]

        success, message = auth_system.register(username, password, face_image, formatted_keystroke_data)
        
        if success:
            return jsonify({"message": message, "success": True})
        else:
            return jsonify({"message": message, "success": False}), 400

    except Exception as e:
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify({"message": f"Registration failed: {str(e)}", "success": False}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        username = request.form['username']
        password = request.form['password']
        face_image = request.form['face_image']
        keystroke_data = json.loads(request.form['keystroke_data'])

        app.logger.info(f"Login attempt for user: {username}")
        app.logger.info(f"Keystroke data received: {keystroke_data}")

        success, message = auth_system.login(username, password, face_image, keystroke_data)
        
        if success:
            session['username'] = username
            app.logger.info(f"Login successful for user: {username}")
            return redirect(url_for('admin_panel'))
        else:
            return jsonify({"message": message, "success": False}), 401

    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({"message": f"Login failed: {str(e)}", "success": False}), 500

@app.route('/admin_panel')
def admin_panel():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    username = session['username']
    # Fetch actual data from your database here
    total_users = 100
    active_users = 50
    failed_logins = 5
    users = [
        {"username": "user1", "last_login": "2023-05-15 14:30:00", "status": "Active"},
        {"username": "user2", "last_login": "2023-05-14 09:15:00", "status": "Inactive"},
        {"username": "user3", "last_login": "2023-05-13 18:45:00", "status": "Active"}
    ]
    
    return render_template('admin_panel.html', 
                           username=username,
                           total_users=total_users, 
                           active_users=active_users, 
                           failed_logins=failed_logins,
                           users=users)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

def fuse_biometric_scores(face_score, keystroke_score, voice_score):
    weights = {'face': 0.4, 'keystroke': 0.3, 'voice': 0.3}
    final_score = (face_score * weights['face'] +
                   keystroke_score * weights['keystroke'] +
                   voice_score * weights['voice'])
    return final_score

if __name__ == '__main__':
    app.run(debug=True)