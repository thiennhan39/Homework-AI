# %%
import os


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from utils import load_conversations, tokenize_and_filter, preprocess_sentence
from model import transformer


# Reproducibility
tf.random.set_seed(1234)
# %%
### Download file
# path_to_zip = tf.keras.utils.get_file(
#     "cornell_movie_dialogs.zip",
#     origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
#     extract=True,
# )


print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))



path_to_dataset = os.path.join(
    os.path.dirname("C:\Assignment\homework_ai\data_ai"), "cornell movie-dialogs corpus"
)

path_to_movie_lines = os.path.join("C:\Assignment\homework_ai\data_ai\cornell_movie_dialogs_extracted\cornell movie-dialogs corpus", "movie_lines.txt")
path_to_movie_conversations = os.path.join("C:\Assignment\homework_ai\data_ai\cornell_movie_dialogs_extracted\cornell movie-dialogs corpus", "movie_conversations.txt")
# %%
# Preprocess file
questions, answers = load_conversations(
    path_to_movie_lines, path_to_movie_conversations, MAX_SAMPLES=999999
)


# %%
# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2 ** 10
)

print("vocab size before:", tokenizer.vocab_size)
# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
print("start token id:", START_TOKEN)
print("end token id:", END_TOKEN)

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2
print("vocab size after:", VOCAB_SIZE)
# %%
# Maximum sentence length
MAX_LENGTH = 40

# Tokenizing and padding
questions, answers = tokenize_and_filter(
    questions, answers, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH
)
# %%
# Prepare TF Dataset
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# decoder inputs use the previous target as input
# remove START_TOKEN from targets

dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"inputs": questions[:], "dec_inputs": answers[:, :-1]},
        answers[:, 1:],
    )
)

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# evaluation
# take 100 samples for evaluating
#eval_encoder_input = questions[:100]
#eval_decoder_input = answers[:100, :-1]
#eval_decoder_output = answers[:100, 1:]
# %%
# Define Transformer model
NUM_LAYERS = 2
D_MODEL = 128
NUM_HEADS = 8
UNITS = 256
DROPOUT = 0.1

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
)
# %%
# Define loss function
def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


# %%
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model_tensor = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  ############################
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model_tensor) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


#%%
# Compile
model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
print(model.summary())

# %%


EPOCHS = 20


history = model.fit(
    dataset,
    epochs=EPOCHS,
    verbose=1,
)

# %%
loss = history.history["loss"]
plt.plot(loss)
plt.xticks(range(0, EPOCHS, 2))
plt.title(f"NUM_LAYERS {NUM_LAYERS}")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

# %%
model.save_weights(f"pretrained_ckpt_len{MAX_LENGTH}_layer{NUM_LAYERS}.weights.h5")
model.load_weights(f"pretrained_ckpt_len{MAX_LENGTH}_layer{NUM_LAYERS}.weights.h5")
model.summary()
# %%
# TEST MODEL
def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0
    )

    output = tf.expand_dims(START_TOKEN, 0)

    for _ in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )

    print("Input: {}".format(sentence))
    print("Output: {}".format(predicted_sentence))

    return predicted_sentence


# %%
# feed the model with its previous output
sentence = "cool ."
for _ in range(5):
    sentence = predict(sentence)
    print("")

# %%
