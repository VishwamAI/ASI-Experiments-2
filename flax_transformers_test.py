from transformers import BertTokenizer, FlaxBertModel
import jax
import jax.numpy as jnp
from flax import linen as nn
from unittest.mock import patch

def test_flax_transformers_compatibility():
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Mock the FlaxBertModel's output
    with patch('transformers.FlaxBertModel.__call__') as mock_model_call:
        mock_model_call.return_value = nn.Module()
        mock_model_call.return_value.last_hidden_state = jnp.zeros((1, 10, 768))

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
