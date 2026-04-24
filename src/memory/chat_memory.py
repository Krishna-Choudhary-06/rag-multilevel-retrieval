class ChatMemory:
    def __init__(self):
        self.store = {}

    def add_message(self, user_id, query, response):
        if user_id not in self.store:
            self.store[user_id] = []

        self.store[user_id].append({
            "query": query,
            "response": response
        })

    def get_context(self, user_id):
        if user_id not in self.store:
            return ""

        history = self.store[user_id][-5:]

        context = ""
        for h in history:
            context += f"User: {h['query']}\nAssistant: {h['response']}\n"

        return context.strip()