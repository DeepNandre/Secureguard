# auth_system.py

from keystroke_model import KeystrokeModel
from database import Database  # Assuming you have a Database class

class SmartAuthenticationSystem:
    def __init__(self):
        self.db = Database()
        self.keystroke_model = KeystrokeModel()
    
    def register(self, username, password, face_image, keystroke_data):
        if self.db.get_user(username):
            return False, "Username already exists"
        
        self.db.add_user(username, password, face_image, keystroke_data)
        return True, "Registration successful"
    
    def login(self, username, password, face_image, keystroke_data):
        user = self.db.get_user(username)
        if user is None:
            return False, "User not found"
        
        if not self.db.verify_password(user['password_hash'], password):
            return False, "Incorrect password"
        
        if not self.verify_face(face_image, user['face_image']):
            return False, "Face verification failed"
        
        if not self.verify_keystroke_dynamics(keystroke_data, user):
            return False, "Keystroke dynamics verification failed"
        
        return True, "Login successful"
    
    def verify_face(self, input_face, stored_face):
        # Implement face verification logic here
        # This is a placeholder and should be replaced with actual face recognition code
        return True
    
    def verify_keystroke_dynamics(self, input_data, stored_data):
        return self.keystroke_model.verify_keystroke(input_data, stored_data['username'])
