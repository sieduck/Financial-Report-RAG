import streamlit as st
import tempfile
import os
from ingestpdf import create_embedding_index
from aiapi import ask
import json
from config import GEMINI_MODEL_LLM, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, TOP_K_FINAL, CONFIDENCE_THRESHOLD
import uuid
from evaluate import parse_golden_dataset, evaluate

st.set_page_config(page_title="Financial Report RAG", page_icon="📈")

st.title("📈 Financial Report Assistant RAG")
st.caption("Upload any company's financial report and ask questions in plain English.")

col1, col2, spacer, col3 = st.columns([0.8, 0.7, 2.5, 1], gap="small")


## Default inits
if "session_id" not in st.session_state:
    # Need completely random session id hence use uuid v4
    st.session_state.session_id = str(uuid.uuid4())
collection_name = f"session_{st.session_state.session_id}"
log_file = f"session_log_{st.session_state.session_id}"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_file" not in st.session_state:
    st.session_state.ingested_file = None
# Init confidence threshold to default
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.7
if "top_k_retrieval" not in st.session_state:
    st.session_state.top_k_retrieval = TOP_K_RETRIEVAL
if "top_k_final" not in st.session_state:
    st.session_state.top_k_final = TOP_K_FINAL
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = CHUNK_SIZE
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = CHUNK_OVERLAP
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = GEMINI_MODEL_LLM
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks=10

st.caption(f"Loaded PDF: {st.session_state.ingested_file}, Number Chunks: {st.session_state.num_chunks}")

@st.dialog("Upload Files")
def upload_dialog():
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if st.button("Done"):
        if uploaded:
            if st.session_state.ingested_file != uploaded.name:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                with st.spinner("Ingesting PDF..."):
                    num_chunks = create_embedding_index(tmp_path, chunk_size=st.session_state.chunk_size,
                                           chunk_overlap=st.session_state.chunk_overlap, collection_name=collection_name)
                st.session_state.ingested_file = uploaded.name
                st.session_state.num_chunks = num_chunks
            st.rerun()


    if st.session_state.ingested_file:
        st.divider()
        st.success(f"{st.session_state.ingested_file} has been uploaded and is ready!")

reload_page = False

with col1:
    with st.popover("Options"):
        with st.form("Configuration"):
            confidence = st.slider(
                "Confidence threshold",
                min_value = 0.000, 
                max_value = 1.000,
                value = st.session_state.confidence_threshold,
                step = 0.001,
                format="%.4f",
                help = "Retrieved chunks minimum relevance score, if higher, chunking model must be confident that chunk contains the answer"
            )

            pdf_loaded = st.session_state.ingested_file is not None

            top_k_r = st.slider(
                "Top K Retrieval", 1, max(2, st.session_state.num_chunks), disabled=not pdf_loaded,
                help="How many K nearest neighbour clusters to pick"
            )
            top_k_f = st.slider(
                "Top K Final (after reranking)", 1, 
                max(2, st.session_state.top_k_retrieval), disabled=not pdf_loaded,
                help="How many K nearest neighbour clusters to pick after reranking"
            )
            chunk_size = st.slider(
                "Chunk Size (tokens)", 100, 1000, st.session_state.chunk_size, step=50,
                help="Size of each chunk"
            )
            chunk_overlap = st.slider(
                "Chunk Overlap", 0, 200, st.session_state.chunk_overlap, step=10,
                help="Allowing chunks to overlap text for smoother transisionss"
            )



            gemini_model = st.selectbox(
                "Gemini Model", 
                ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
                index=["gemini-2.5-flash", "gemini-2.5-flash-lite"].index(st.session_state.gemini_model)
            )

            submitted = st.form_submit_button("Apply Settings")

            if submitted:
                # Check if changed chunk settings
                if (chunk_size != st.session_state.chunk_size) or (chunk_overlap != st.session_state.chunk_overlap):
                    st.session_state.ingested_file = None
                    reload_page = True

                st.session_state.confidence_threshold = confidence
                st.session_state.top_k_retrieval = top_k_r
                st.session_state.top_k_final = top_k_f
                st.session_state.chunk_size = chunk_size
                st.session_state.chunk_overlap = chunk_overlap
                st.session_state.gemini_model = gemini_model
                
                st.session_state.settings_changed = True


if st.session_state.get("settings_changed") == True:
    st.toast("Settings successfully changed")
    st.session_state.settings_changed = False
    if reload_page == True:
        st.toast("Chunking settings have been changed. To apply, reupload PDF")



with col2:
    if st.button("Clear files"):
        st.session_state.ingested_file = None
        st.session_state.num_chunks = 0
        st.rerun()

with col3:
    if st.button("📁 Upload Files"):
        upload_dialog()


chat_tab, stats_tab, evaluation_tab = st.tabs(["💬 Chat", "📊 Stats", "🤔 Evaluation"])        

with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the financial report..."):
        if not st.session_state.ingested_file:
            st.error("Please upload a PDF first.")
        else:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask(prompt, st.session_state.confidence_threshold, 
                                 st.session_state.top_k_retrieval, st.session_state.top_k_final, 
                                 st.session_state.gemini_model, collection_name=collection_name, 
                                 log_name=log_file
                                 )
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

with stats_tab:
    st.subheader("RAG statistics")

    try:
        with open(f"{log_file}.json") as f:
            logs = json.load(f)

        if not logs:
            st.info("No queries logged yet. Ask some questions first.")
        else:
            total_queries = len(logs)
            avg_retrieval = sum(l["retrieval_time_sec"] for l in logs) / total_queries
            avg_generation = sum(l["generation_time_sec"] for l in logs) / total_queries
            avg_total = sum(l["total_time_sec"] for l in logs) / total_queries

            # metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Queries", total_queries)
            c2.metric("Avg Retrieval Time", f"{avg_retrieval:.2f}s")
            c3.metric("Avg Generation Time", f"{avg_generation:.2f}s")
            c4.metric("Avg Total Time", f"{avg_total:.2f}s")

            # query history table
            st.divider()
            st.subheader("Query History")

            table_data = []

            for log in reversed(logs):
                timestamp = log["timestamp"][:19]
                question = log["question"]

                # If there is a cutoff in the length of the questino
                if len(question) > 60:
                    question = question[:60] + "..."


                row = {
                    "Time": timestamp,
                    "Question": question,
                    "Retrieval (s)": log["retrieval_time_sec"],
                    "Generation (s)": log["generation_time_sec"],
                }

                table_data.append(row)

            st.dataframe(table_data, width="stretch")
    except FileNotFoundError:
        st.info("Empty. Please run some queries")

with evaluation_tab:
    st.subheader("Evaluation")
    st.caption("Upload golden dataset in any format to evaluate the RAG system. Golden dataset can be of any question answer format.")


    if not st.session_state.ingested_file:
        st.warning("Please upload PDF via 📁 Upload Files")
    else: 
        file_content = st.file_uploader("Upload Golden Dataset", type=["json", "csv", "txt"])

        if file_content:
            content = file_content.read().decode("utf-8")
            st.success(f"Loaded: {file_content.name}")

            if st.button("Run Evaluation Via Golden Dataset"):
                with st.spinner("Converting dataset into json format via Gemini"):
                    try:
                        golden_dataset = parse_golden_dataset(content)
                        st.info(f"Found {len(golden_dataset)} question answer pairs")

                    except Exception as e:
                        st.error(f"Error: {e}. Failed to parse dataset")
                        golden_dataset = []
                
                if golden_dataset:
                    with st.spinner(f"Evaluating {len(golden_dataset)} questions"):
                        results = evaluate(
                            golden_dataset, confidence_threshold=st.session_state.confidence_threshold,
                            top_k_retrieval=st.session_state.top_k_retrieval, top_k_final=st.session_state.top_k_final,
                            gemini_model_llm=st.session_state.gemini_model, collection_name=collection_name,
                            log_name=log_file
                        )
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Number questions evaluated", results["total_questions"])
                    c2.metric("Average score", f"{results["average_score"]:.2f}/5")
                    c3.metric("Overall Accuracy", f"{(results["average_score"] / 5) * 100:.2f}%")

                    st.divider()
                    st.subheader("Per question results")
                    st.dataframe(results["results"])




