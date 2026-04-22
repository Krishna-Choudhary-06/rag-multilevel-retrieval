class ChatMemory:
    def __init__(self):
        # store memory per user
        self.store = {}

    # =========================
    # ADD MESSAGE
    # =========================
    def add_message(self, user_id, query, response):
        if user_id not in self.store:
            self.store[user_id] = []

        self.store[user_id].append({
            "query": query,
            "response": response
        })

    # =========================
    # GET CONTEXT (FIXED)
    # =========================
    def get_context(self, user_id):
        if user_id not in self.store:
            return ""

        history = self.store[user_id][-5:]  # last 5 messages

        context = ""
        for item in history:
            context += f"User: {item['query']}\n"
            context += f"Assistant: {item['response']}\n"

        return context.strip()