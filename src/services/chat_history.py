from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return StreamlitChatMessageHistory(
        key=f"chat_history_{session_id}"
    )