# Code for pretraining the BLIP2 Q-former model

This repo contains the code for pretraining the BLIP2 Q-former model. The code is modified from the [Hugging Face Transformers](https://github.com/huggingface/transformers/blob/82486e5995ed0a65520b10ce1ea938214a199231/src/transformers/models/blip_2/modeling_blip_2.py#L930).
The original hugging face implementation does not support pretraining. We have added the pretraining code to the original implementation.

- blip_2_m.py: Modified from the original modeling_blip_2.py from Hugging Face Transformers. The Blip2QFormerModel class in this file is modified to support pretraining.
- blip_2_pretrain.py: Contains the pretrain model class BLIP2QformerPretrain for pretraining the q-former model. A visual encoder should be provided to the model to connect with the Q-former. This code uses the LAVIS/lavis/models/blip2_models/blip2_qformer.py from Salesforce as a reference but modified to suport the hugging face implementation.
- pretrain.py: Main script for pretraining.
