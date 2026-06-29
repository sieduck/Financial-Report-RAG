# Models used by the app
GEMINI_MODEL_LLM = "gemini-2.0-flash"
GEMINI_MODEL_EMBEDDING = "gemini-embedding-001"


# Chunking setup
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Retrieval setup
# Grab a few extra chunks first, then rerank down to the best ones
TOP_K_RETRIEVAL = 10
TOP_K_FINAL = 3

# Threshold of confidence to which we accept a chunk
CONFIDENCE_THRESHOLD = 0.0001


# Where query logs get saved
LOG_FILE = "monitor_log.json"