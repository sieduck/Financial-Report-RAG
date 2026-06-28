import streamlit as st
import tempfile
import os
from ingestpdf import create_embedding_index
from aiapi import ask
import json

st.set_page_config(page_title="Financial Report RAG", page_icon="📈")

st.title("📈 Financial Report Assistant RAG")
st.caption("Upload any company's financial report and ask questions in plain English.")

col1, col2, spacer, col3 = st.columns([0.8, 0.7, 2.5, 1], gap="small")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_file" not in st.session_state:
    st.session_state.ingested_file = None

# Init confidence threshold to default
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.7

st.caption(f"Loaded PDF: {st.session_state.ingested_file}")

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
                    create_embedding_index(tmp_path)
                st.session_state.ingested_file = uploaded.name
            st.rerun()


    if st.session_state.ingested_file:
        st.divider()
        st.success(f"{st.session_state.ingested_file} has been uploaded and is ready!")

    

with col1:
    with st.popover("Options"):
        st.session_state.confidence_threshold = st.slider(
            "Confidence threshold",
            min_value = 0.000, 
            max_value = 1.000,
            value = 0.750,
            step = 0.001,
            format="%.4f",
            help = "Retrieved chunks minimum relevance score, if higher, chunking model must be confident that chunk contains the answer"
        )

with col2:
    if st.button("Clear files"):
        st.session_state.ingested_file = None
        st.rerun()

with col3:
    if st.button("📁 Upload Files"):
        upload_dialog()


chat_tab, stats_tab = st.tabs(["💬 Chat", "📊 Stats"])        

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
                    answer = ask(prompt, st.session_state.confidence_threshold)
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

with stats_tab:
    st.subheader("RAG statistics")

    try:
        with open("monitor_log.json") as f:
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