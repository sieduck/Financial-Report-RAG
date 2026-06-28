import json
import time
from datetime import datetime

LOG_FILE = "monitor_log.json"

def log_query(question, answer, chunks, retrieval_time, generation_time, input_tokens, output_tokens):
    json_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "chunks_retrieved": chunks,
        "retrieval_time_sec": round(retrieval_time, 3),
        "generation_time_sec": round(generation_time, 3),
        "total_time_sec": round(retrieval_time + generation_time, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    # Check if log file already exists
    try:
        with open(LOG_FILE) as f:
            log = json.load(f)
    except:
        log = []

    log.append(json_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=1)