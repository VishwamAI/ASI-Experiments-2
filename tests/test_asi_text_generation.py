import unittest
import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.text_generation import TextGeneration

class TestTextGeneration(unittest.TestCase):
    def setUp(self):
        self.text_gen = TextGeneration()

    def test_generate_text_default(self):
        prompt = "Once upon a time"
        generated_texts = self.text_gen.generate_text(prompt)
        self.assertIsInstance(generated_texts, list)
        self.assertGreater(len(generated_texts), 0)
        self.assertIsInstance(generated_texts[0], str)

    def test_generate_text_custom_length(self):
        prompt = "In a galaxy far, far away"
        max_length = 100
        generated_texts = self.text_gen.generate_text(prompt, max_length=max_length)
        self.assertIsInstance(generated_texts, list)
        self.assertGreater(len(generated_texts), 0)
        self.assertIsInstance(generated_texts[0], str)
        self.assertLessEqual(len(generated_texts[0].split()), max_length)

    def test_generate_text_multiple_sequences(self):
        prompt = "The quick brown fox"
        num_return_sequences = 3
        generated_texts = self.text_gen.generate_text(prompt, num_return_sequences=num_return_sequences, num_beams=3)
        self.assertIsInstance(generated_texts, list)
        self.assertEqual(len(generated_texts), num_return_sequences)
        for text in generated_texts:
            self.assertIsInstance(text, str)

if __name__ == "__main__":
    unittest.main()
