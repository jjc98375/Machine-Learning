import sys
from unittest.mock import MagicMock

# MOCKING DEPENDENCIES BEFORE IMPORT
# This allows us to test the logic even if torch/transformers are not installed
mock_torch = MagicMock()
mock_torch.tensor = MagicMock(side_effect=lambda x: x) # Mock tensor conversion to return list/original
sys.modules['torch'] = mock_torch

mock_utils_data = MagicMock()
mock_utils_data.Dataset = object # Ensure Dataset is a standard class, not a Mock, so __getitem__ runs normally
sys.modules['torch.utils.data'] = mock_utils_data

sys.modules['datasets'] = MagicMock()
sys.modules['transformers'] = MagicMock()

import unittest
# Now import the module to test
from src.dataset import SwitchLinguaDataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Mock the load_dataset call within the class
        # We need to patch the global load_dataset imported in src.dataset, 
        # or just rely on the fact that we mocked 'datasets' module.
        # Since 'from datasets import load_dataset' is in src/dataset.py, 
        # and we mocked 'datasets' before import, 'load_dataset' in src/dataset 
        # is already a Mock object.
        
        # We can configure the mock to return data
        self.mock_data = [
            {
                'tokens': ['I', 'am', 'going', 'to', 'la', 'casa'],
                'lid': ['en', 'en', 'en', 'en', 'es', 'es']
            }
        ]
        
        # We also need to mock AutoTokenizer
        # In src/dataset.py: from transformers import AutoTokenizer
        # AutoTokenizer is a Mock.
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': MagicMock(),
            'attention_mask': MagicMock(),
            'word_ids': lambda: [0, 1, 2, 3, 4, 5] # One-to-one mapping for simplicity in this test
        }
        # Be careful: encoding['input_ids'] is accessed.
        # encoding.word_ids() is called.
        
        # We need to configure the AutoTokenizer on the mocked transformers module
        # Since we patched sys.modules['transformers'], we can access it via key or import
        transformers = sys.modules['transformers']
        transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        self.mock_tokenizer_instance = mock_tokenizer_instance

    def test_logic_flow(self):
        # Instantiating the dataset
        ds = SwitchLinguaDataset()
        # Inject our mock data into the dataset instance
        ds.dataset = self.mock_data
        
        # We also need to ensure tokenizer is set (it is set in __init__)
        # but we already mocked it.
        
        # Configure the tokenizer mock for the specific call in __getitem__
        # "tokens[:-1]" are passed.
        # We want word_ids to return [0, 1, 2, 3, 4] (indices into the NEW trimmed tokens)
        # The trimmed tokens are ['I', 'am', 'going', 'to', 'la']
        # Their Original indices were 0, 1, 2, 3, 4.
        
        shape_mock = MagicMock()
        shape_mock.squeeze.return_value = "tensor"
        
        # Helper to behave like BatchEncoding (dict-like + methods)
        encoding_mock = MagicMock()
        data_dict = {'input_ids': shape_mock, 'attention_mask': shape_mock}
        encoding_mock.__getitem__.side_effect = data_dict.__getitem__
        encoding_mock.word_ids.return_value = [0, 1, 2, 3, 4]
        
        self.mock_tokenizer_instance.return_value = encoding_mock

        # CALL __getitem__
        item = ds[0]
        
        # Verify Labels
        # Logic check:
        # 0: 'I' -> next 'am' (en) -> 0
        # 1: 'am' -> next 'going' (en) -> 0
        # 2: 'going' -> next 'to' (en) -> 0
        # 3: 'to' -> next 'la' (es) -> SWITCH=1. Duration 'la', 'casa' = 2 -> Class 0.
        # 4: 'la' -> next 'casa' (es) -> No Switch (0).
        
        # Expected switch labels: [0, 0, 0, 1, 0]
        # Expected duration labels: [-100, -100, -100, 0, -100]
        
        print("Generated Labels:", item['labels_switch'])
        
        self.assertEqual(item['labels_switch'], [0, 0, 0, 1, 0])
        self.assertEqual(item['labels_duration'], [-100, -100, -100, 0, -100])

if __name__ == '__main__':
    unittest.main()
