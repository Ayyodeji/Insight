from transformers import DPRQuestionEncoder, DPRContextEncoder

# Define paths to the pretrained DPR models and tokenizers
context_encoder_path = 'facebook/dpr-ctx_encoder-multiset-base'
question_encoder_path = 'facebook/dpr-question_encoder-multiset-base'

# Initialize the DPR question and context encoders
question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_path)
context_encoder = DPRContextEncoder.from_pretrained(context_encoder_path)

# Save the question and context encoders to a directory
question_encoder.save_pretrained("./models/question_encoder")
context_encoder.save_pretrained("./models/context_encoder")
