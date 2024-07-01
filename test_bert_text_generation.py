import unittest
from bert_text_generation import BERTTextGeneration

class TestBERTTextGeneration(unittest.TestCase):
    def setUp(self):
        self.text_gen = BERTTextGeneration()

    def test_generate_text_single_sequence(self):
        prompt = "Once upon a time"
        result = self.text_gen.generate_text(prompt, max_length=20, num_return_sequences=1)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)
        self.assertGreater(len(result[0]), len(prompt))

    def test_generate_text_multiple_sequences(self):
        prompt = "In a galaxy far, far away"
        result = self.text_gen.generate_text(prompt, max_length=20, num_return_sequences=3)
        self.assertEqual(len(result), 3)
        for text in result:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), len(prompt))

if __name__ == '__main__':
    unittest.main()
