import unittest
from flask import Flask
from backend_flask import app  

class TestFlaskApp(unittest.TestCase):

    def setUp(self):
        # Set up a test client
        self.app = app.test_client()

    def test_load_model_endpoint(self):
        # Test the /load_model endpoint
        response = self.app.get('/load_model')
        data = response.get_json()

        # Assert that the response contains the model_path key
        self.assertIn('model_path', data)

    def test_full_dataframe_endpoint(self):
        # Test the /full_dataframe endpoint
        response = self.app.get('/full_dataframe')
        data = response.get_json()

        # Assert that the response is a list of dictionaries
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(entry, dict) for entry in data))

    def test_original_data_endpoint(self):
        # Test the /original_data endpoint
        response = self.app.get('/original_data')
        data = response.get_json()

        # Assert that the response is a list of dictionaries
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(entry, dict) for entry in data))

    def test_get_client_list_endpoint(self):
        # Test the /get_client_list endpoint
        response = self.app.get('/get_client_list')
        data = response.get_json()

        # Assert that the response is a list
        self.assertIsInstance(data, list)

    def test_predict_endpoint(self):
        # Test the /predict endpoint
        sample_data = {"feature1": 1, "feature2": 2}  # Replace with actual feature names and values
        response = self.app.post('/predict', json=sample_data)
        data = response.get_json()

        # Assert that the response contains the prediction key
        self.assertIn('prediction', data)

if __name__ == '__main__':
    unittest.main()
