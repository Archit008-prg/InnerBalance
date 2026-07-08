from django.apps import AppConfig
import os
import threading

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        # Only run pre-warming in the main worker process (ignore reloader)
        if os.environ.get('RUN_MAIN') == 'true':
            def pre_warm():
                try:
                    print("[LLM PRE-WARM] Starting asynchronous LLM pre-warming...")
                    from api.rag_views import rag_system
                    # Accessing the .llm property triggers model/pipeline loading
                    _ = rag_system.llm
                    print("[LLM PRE-WARM] LLM pre-warmed successfully!")
                except Exception as e:
                    print(f"[LLM PRE-WARM ERROR] Asynchronous pre-warming failed: {e}")

            threading.Thread(target=pre_warm, daemon=True).start()
