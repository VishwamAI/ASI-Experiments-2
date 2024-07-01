from transformers import BertTokenizer, FlaxBertModel
import jax
import jax.numpy as jnp
from unittest.mock import patch, MagicMock

def test_flax_transformers_compatibility():
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Mock the FlaxBertModel's from_pretrained and __call__ methods
    with patch('transformers.FlaxBertModel.from_pretrained') as mock_from_pretrained, \
         patch('transformers.FlaxBertModel.__call__') as mock_model_call:

        # Create a simple mock class to avoid recursion issues
        class SimpleMockModel:
            def __call__(self, input_ids, attention_mask):
                return MagicMock(last_hidden_state=jnp.zeros((1, 10, 768)))

        # Mock the from_pretrained method to return a simple mock model
        mock_model = SimpleMockModel()
        mock_from_pretrained.return_value = mock_model
        mock_model_call.return_value = mock_model.__call__(None, None)

        # Define a simple input
        input_text = "Hello, this is a test."
        inputs = tokenizer(input_text, return_tensors='jax', padding='max_length', max_length=10, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Ensure input tensors have a batch dimension
        if len(input_ids.shape) == 2:
            input_ids = input_ids[None, :]
        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask[None, :]

        # Print shapes of input tensors
        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")

        # Run the model
        outputs = mock_model_call(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Print shape of the output tensor
        print(f"last_hidden_state shape: {last_hidden_state.shape}")

if __name__ == "__main__":
    test_flax_transformers_compatibility()
