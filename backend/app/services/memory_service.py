conversation_store = {}


def add_message(session_id: str, role: str, content: str):

    if session_id not in conversation_store:
        conversation_store[session_id] = []

    conversation_store[session_id].append(
        {
            "role": role,
            "content": content
        }
    )


def get_history(session_id: str):

    if session_id not in conversation_store:
        conversation_store[session_id] = []

    return conversation_store[session_id]