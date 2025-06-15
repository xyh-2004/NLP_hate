PRETRAINED_MODEL = "bert-base-chinese"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 3e-5

TARGETED_GROUPS = ["Region", "Racism", "Sexism", "LGBTQ", "others", "non-hate"]

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
