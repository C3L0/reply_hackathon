import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langfuse import observe

class KnowledgeBaseAgent:
    def __init__(self):
        # Utilisation d'un modèle d'embedding léger et performant
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self._initialize_base_knowledge()

    def _initialize_base_knowledge(self):
        """Initialise la base avec les typologies de fraude connues."""
        initial_patterns = [
            "Vishing (Voice Phishing): Appel suspect simulant un proche (petit-fils, accident) avec une voix synthétique sans bruit de fond, suivi d'un virement immédiat.",
            "Phishing Classique: Réception d'un SMS ou Mail urgent (compte bloqué, amende) contenant un lien suspect, suivi d'une connexion inhabituelle.",
            "Impossible Travel: Transaction physique effectuée dans une ville éloignée de la dernière position GPS connue dans un laps de temps trop court.",
            "Fraude au Président: Email usurpant l'identité d'un cadre demandant un virement confidentiel et urgent vers un nouvel IBAN.",
            "E-commerce Senior: Achat de gros montant sur un site e-commerce inhabituel par un profil de plus de 75 ans tard dans la nuit.",
            "High Velocity Outlier: Série de petites transactions rapides vidant progressivement le compte, souvent après un vol de credentials."
        ]
        
        documents = [Document(page_content=text, metadata={"source": "initial_knowledge"}) for text in initial_patterns]
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    @observe()
    def retrieve_context(self, query: str, k: int = 2) -> str:
        """Récupère les archétypes de fraude les plus proches de la requête."""
        if not self.vector_store:
            return ""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        context = "Context: The following historical fraud patterns match the current situation:\n"
        for doc, score in results:
            if score < 1.5: # Seuil de similarité (plus bas est meilleur pour FAISS L2)
                context += f"- {doc.page_content}\n"
        return context

    @observe()
    def learn_new_pattern(self, description: str):
        """Ajoute un nouveau pattern détecté à la base de connaissance (Self-Updating)."""
        doc = Document(page_content=description, metadata={"source": "autonomous_learning"})
        self.vector_store.add_documents([doc])
        print(f"Knowledge Base Updated: Learned new pattern.")
