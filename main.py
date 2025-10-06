import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.constants.llm import llm
from src.services.pdf_processor import process_pdf
from src.services.vector_store import create_vector_store, save_vector_store, load_vector_store
from src.services.chat_history import get_session_history

# Page configuration
st.set_page_config(page_title="PaperPal - Research Assistant", page_icon="📄")
st.title("📄 PaperPal - Research Paper Chat Assistant")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_123"

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = None

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload Research Paper")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_paper.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                # Process PDF
                documents = process_pdf("temp_paper.pdf")
                
                # Create vector store
                st.session_state.vector_store = create_vector_store(documents)
                save_vector_store(st.session_state.vector_store)
                
                st.success(f"PDF processed! {len(documents)} chunks created.")

# Create conversational RAG chain
if st.session_state.vector_store is not None and st.session_state.conversational_chain is None:
    
    # Create retriever
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Contextualize question prompt
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )
    
    # Answer question prompt
    qa_system_prompt = """You are a helpful research assistant. \
    Answer the following question based on the provided context from the research paper. \
    Think step by step before providing a detailed answer. \
    
    IMPORTANT FORMATTING RULES:
    1. For mathematical equations, matrices, and formulas, use proper Markdown/LaTeX formatting
    2. For matrices, use this format:
    K^T =
    | 7 10 13 |
    | 8 11 14 |
    | 9 12 15 |

    3. For inline math, use this format: `x = y + z`
    4. For complex equations, break them down step by step with proper spacing
    5. Use bullet points or numbered lists for step-by-step explanations
    6. Always present matrices in a readable grid format with proper alignment
    
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )
    
    # Wrap with message history
    st.session_state.conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

# Display chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the research paper..."):
    if st.session_state.conversational_chain is None:
        st.warning("Please upload and process a PDF first!")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversational_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                
                answer = response["answer"]
                st.markdown(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    get_session_history(st.session_state.session_id).clear()
    st.rerun()
