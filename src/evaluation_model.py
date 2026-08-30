from deepeval.models import OllamaModel

evaluation_model = OllamaModel(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0,
)
