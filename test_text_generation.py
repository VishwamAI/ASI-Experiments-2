import unittest
from text_generation import TextGeneration

class TestTextGeneration(unittest.TestCase):
    def setUp(self):
        self.text_gen = TextGeneration()

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

    def test_generate_text_with_beams(self):
        prompt = "The quick brown fox"
        result = self.text_gen.generate_text(prompt, max_length=20, num_return_sequences=1, num_beams=3)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)
        self.assertGreater(len(result[0]), len(prompt))

if __name__ == '__main__':
    unittest.main()
