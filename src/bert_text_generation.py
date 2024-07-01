from transformers import FlaxBertModel, BertTokenizer
import jax
import jax.numpy as jnp
from flax.training.common_utils import onehot

class BERTTextGeneration:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = FlaxBertModel.from_pretrained(model_name)
        self.config = self.model.config

        # Print model configuration parameters
        print(f"num_attention_heads: {self.config.num_attention_heads}")
        print(f"head_dim: {self.config.hidden_size // self.config.num_attention_heads}")

    def generate_text(self, prompt, max_length=50, num_return_sequences=1):
        inputs = self.tokenizer(prompt, return_tensors='jax', padding='max_length', max_length=max_length, truncation=True)
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

        def model_forward(params, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, params=params)
            # Print shape of last_hidden_state tensor
            print(f"last_hidden_state shape: {outputs.last_hidden_state.shape}")
            return outputs.last_hidden_state

        params = self.model.params
        outputs = jax.vmap(model_forward, in_axes=(None, 0, 0), out_axes=0)(params, input_ids, attention_mask)

        # Print shape of output tensor
        print(f"outputs shape: {outputs.shape}")

        # Adjust the handling of the model's output to match the expected usage
        generated_texts = []
        for i in range(num_return_sequences):
            # Use the corresponding batch element
            output_sequence = outputs[i, :, :]
            # Print shape of output sequence before decoding
            print(f"output_sequence shape: {output_sequence.shape}")
            # Decode the output sequence after flattening
            generated_text = self.tokenizer.decode(jnp.argmax(output_sequence, axis=-1).flatten(), skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts
