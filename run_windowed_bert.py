import logging
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", "true"],
    ["Frodo was the heir of Isildur", "false"],
]
train_path = "data/product_augmented/back/emanual_balanced_review_hijack_train.tsv"
test_path = "data/product_augmented/back/emanual_balanced_review_hijack_test.tsv"
val_path = "data/product_augmented/back/emanual_balanced_review_hijack_val.tsv"
train_df = pd.read_csv(train_path)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", "true"],
    ["Merry was the king of Rohan", "false"],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs=1
model_args.labels_list = ["true", "false"]

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])