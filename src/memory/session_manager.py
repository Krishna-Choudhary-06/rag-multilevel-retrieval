class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get_pipeline(self, user_id):
        from src.pipeline.rag_pipeline import RAGPipeline

        if user_id not in self.sessions:
            print(f"[SESSION] Creating new session for {user_id}")
            self.sessions[user_id] = RAGPipeline()

        return self.sessions[user_id]

    def clear_sessions(self):
        print("[SESSION] Clearing all cached pipelines")
        self.sessions = {}
