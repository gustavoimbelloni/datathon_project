import unittest
import pandas as pd
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import preprocess_data

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Criar um dataframe de exemplo
        self.sample_data = pd.DataFrame({
            'Fase': [1, 2],
            'Idade 22': [10, 12],
            'Gênero': ['M', 'F'],
            'Ano ingresso': [2020, 2021],
            'IAA': [8.0, 7.5],
            'IEG': [9.0, 8.0],
            'IPS': [7.0, 8.5],
            'IDA': [8.5, 7.0],
            'Matem': [8.0, 7.0],
            'Portug': [7.5, 8.0],
            'IPV': [8.0, 7.5],
            'IAN': [9.0, 8.0],
            'Defas': [-1, 0]
        })

    def test_preprocess_data_columns(self):
        processed = preprocess_data(self.sample_data)
        self.assertIn('target', processed.columns)
        self.assertNotIn('Defas', processed.columns)
        self.assertEqual(processed['target'].iloc[0], 1) # Defas -1 -> target 1
        self.assertEqual(processed['target'].iloc[1], 0) # Defas 0 -> target 0

    def test_gender_encoding(self):
        processed = preprocess_data(self.sample_data)
        self.assertEqual(processed['Gênero'].iloc[0], 1) # M -> 1
        self.assertEqual(processed['Gênero'].iloc[1], 0) # F -> 0

if __name__ == '__main__':
    unittest.main()
