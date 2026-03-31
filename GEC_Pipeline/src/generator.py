import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class GECGenerator:
    def __init__(self, model_name="llama-3.1-8b-instant", temperature=0.5):
        # This automatically looks for GROQ_API_KEY in your environment/.env
        self.llm = ChatGroq(model_name=model_name, temperature=temperature)
        
        self.prompt = PromptTemplate(
            input_variables=["sentence", "error_type"],
            template="Rewrite this sentence to include exactly one {error_type}. "
                    "Output ONLY the corrupted sentence.\n\nSentence: {sentence}"
        )

    def corrupt_sentence(self, sentence, error_type):
        """Uses the LLM to inject a specific error."""
        formatted_prompt = self.prompt.format(sentence=sentence, error_type=error_type)
        response = self.llm.invoke(formatted_prompt)
        return response.content.strip()