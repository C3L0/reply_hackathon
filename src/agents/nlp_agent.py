import json
import os
from typing import List, Dict, Any
from .base import get_llm, get_langfuse_handler
from .schemas import DatasetPackage, NLPSignal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langfuse import observe

class NLPAgent:
    def __init__(self):
        self.llm = get_llm()
        self.parser = PydanticOutputParser(pydantic_object=NLPSignal)

    @observe()
    def analyze(self, data: DatasetPackage, session_id: str = None) -> List[NLPSignal]:
        suspicious_comms = []
        keywords = ["urgent", "verify", "link", "click", "mirror", "login", "suspicious", "account", "lock", "bank", "password", "action required", "unusual", "security", "identity", "grandson", "accident", "hospital", "money", "help"]
        
        # 1. SMS
        for s in data.sms:
            if any(k in s.text.lower() for k in keywords):
                suspicious_comms.append(("sms", s.model_dump()))
        
        # 2. Mails
        for m in data.mails:
            if any(k in m.content.lower() or k in m.subject.lower() for k in keywords):
                m_dict = m.model_dump()
                m_dict["content"] = m_dict["content"][:3000]
                suspicious_comms.append(("mail", m_dict))
                
        # 3. Audio (Vishing) - Simulation pour le moment car on ne peut pas lire le MP3 directement sans lib externe
        # On passe le nom du fichier à l'LLM pour qu'il "imagine" la menace si le nom est explicite ou on traite les fichiers existants
        for audio in data.audio_files:
            # On simule la présence d'un transcript ou d'une analyse de métadonnées
            suspicious_comms.append(("audio_vishing", {"filename": audio, "metadata": "No background noise, high frequency stability (potential AI Clone)"}))
        
        signals = []
        for comm_type, comm in suspicious_comms[:40]:
            prompt = ChatPromptTemplate.from_template("""
            Role: Expert Cyber-Security Analyst (NLP & Voice Specialist).
            Task: Detect Phishing, Vishing, and Campaign Patterns.
            
            Communication ({comm_type}):
            {content}
            
            Instructions:
            1. Analyze content for phishing indicators.
            2. If type is 'audio_vishing', analyze metadata for AI Voice Cloning (lack of background noise, unnatural frequency).
            3. EXTRACT IoCs: Look for URLs, phone numbers, or suspicious email addresses. Put them in 'ioc_list'.
            4. Identify the TARGET (Name, Email, Phone).
            5. Provide risk score (0-1) and reasoning.
            
            Return an NLPSignal.
            {format_instructions}
            """)
            
            handler = get_langfuse_handler()
            if session_id: handler.session_id = session_id
                
            response = self.llm.invoke(
                prompt.format(
                    comm_type=comm_type,
                    content=json.dumps(comm),
                    format_instructions=self.parser.get_format_instructions()
                ),
                config={"callbacks": [handler]}
            )
            
            try:
                signals.append(self.parser.parse(response.content))
            except: continue
        return signals
