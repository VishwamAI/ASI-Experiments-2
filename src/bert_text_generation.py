from transformers import FlaxBertModel, BertTokenizer
import jax
import jax.numpy as jnp
from flax.training.common_utils import onehot

class BERTTextGeneration:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = FlaxBertModel.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=50, num_return_sequences=1):
        inputs = self.tokenizer(prompt, return_tensors='jax', padding='max_length', max_length=max_length, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")

        def model_forward(params, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, params=params).last_hidden_state

        params = self.model.params
        outputs = jax.vmap(model_forward, in_axes=(None, 0, 0))(params, input_ids, attention_mask)

        print(f"outputs shape: {outputs.shape}")

        # Adjust the handling of the model's output to match the expected usage
        generated_texts = []
        for i in range(num_return_sequences):
            generated_text = self.tokenizer.decode(outputs[0][i], skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts
