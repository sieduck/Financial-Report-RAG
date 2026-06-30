import json
import time
from datetime import datetime


def log_query(question, answer, chunks, retrieval_time, generation_time, input_tokens, output_tokens, log_name="monitor_log"):
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
        with open(f"{log_name}.json") as f:
            log = json.load(f)
    except:
        log = []

    log.append(json_entry)

    with open(f"{log_name}.json", "w") as f:
        json.dump(log, f, indent=1)