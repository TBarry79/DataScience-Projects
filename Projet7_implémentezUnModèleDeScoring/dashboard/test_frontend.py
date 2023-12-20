import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from io import StringIO

# Import the functions you want to test
from frontend_streamlit import (
    get_client_list,
    get_dataframe_from_api,
    get_original_data,
    get_prediction,
    load_model_from_backend,
    create_gauge,
)

class TestYourStreamlitApp(unittest.TestCase):

    @patch('requests.get')
    def test_get_client_list(self, mock_requests_get):
        # Mock the requests.get function to return a specific response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["client1", "client2"]  # Replace with your expected result
        mock_requests_get.return_value = mock_response

        # Call the function and check the result
        result = get_client_list()
        self.assertEqual(result, ["client1", "client2"])

    @patch('requests.get')
    def test_get_dataframe_from_api(self, mock_requests_get):
        # Mock the requests.get function to return a specific response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"column1": [1, 2], "column2": ["a", "b"]}'  # Replace with your expected result
        mock_requests_get.return_value = mock_response

        # Call the function and check the result
        result = get_dataframe_from_api("http://example.com")
        expected_df = pd.DataFrame({"column1": [1, 2], "column2": ["a", "b"]})
        pd.testing.assert_frame_equal(result, expected_df)

    # Add similar tests for other functions...

if __name__ == '__main__':
    unittest.main()
