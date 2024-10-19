import unittest
from app import app, auth_system
import json
import os
from PIL import Image
import io

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True 
        # Create a dummy image for testing
        self.test_image_path = 'test_image.jpg'
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(self.test_image_path)
        
        # Ensure a fresh database for each test
        if os.path.exists(auth_system.db_path):
            os.remove(auth_system.db_path)
        auth_system.create_user_table()

    def tearDown(self):
        # Remove the test image after tests
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        if os.path.exists(auth_system.db_path):
            os.remove(auth_system.db_path)

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_register(self):
        with open(self.test_image_path, 'rb') as test_image:
            data = {
                'username': 'testuser@example.com',
                'password': 'testpassword',
                'face_image': (test_image, 'test_image.jpg')
            }
            response = self.app.post('/register', 
                                     data=data, 
                                     content_type='multipart/form-data')
            
            print(f"Register response status: {response.status_code}")
            print(f"Register response data: {response.data.decode('utf-8')}")
            
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"User registered successfully", response.data)

        # Check if the user was actually created in the database
        with app.app_context():
            conn = auth_system.get_db_connection()
            user = conn.execute('SELECT * FROM users WHERE username = ?', ('testuser@example.com',)).fetchone()
            print(f"User in database after registration: {user is not None}")
            if user:
                print(f"User data: {dict(user)}")
            else:
                print("User not found in database")

    def test_login(self):
        self.test_register()
        
        login_data = {
            'username': 'testuser@example.com',
            'password': 'testpassword'
        }
        response = self.app.post('/login', data=login_data)
        
        print(f"Login response status: {response.status_code}")
        print(f"Login response data: {response.data.decode('utf-8')}")
        
        # Check the database again after login attempt
        with app.app_context():
            conn = auth_system.get_db_connection()
            user = conn.execute('SELECT * FROM users WHERE username = ?', ('testuser@example.com',)).fetchone()
            print(f"User in database after login attempt: {user is not None}")
            if user:
                print(f"User data: {dict(user)}")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Login successful", response.data)

    def test_authenticate(self):
        self.test_login()  # This will register and log in a user
        
        # Simulate keystroke data
        keystroke_data = [100, 200, 300, 400, 500]  # Example keystroke timing data
        
        response = self.app.post('/authenticate', 
                                 json={'keystroke_data': keystroke_data},
                                 content_type='application/json')
        
        print(f"Authenticate response status: {response.status_code}")
        print(f"Authenticate response data: {response.data.decode('utf-8')}")
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('auth_score', response_data)
        self.assertIn('risk_level', response_data)

if __name__ == '__main__':
    unittest.main()