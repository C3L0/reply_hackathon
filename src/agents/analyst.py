import json
import pandas as pd
from typing import List, Dict
from .base import get_llm, get_langfuse_handler
from .schemas import FlaggedTransaction
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe

class AnalystAgent:
    def __init__(self):
        self.llm = get_llm()
        
    @observe()
    def analyze(self, flagged_tx: List[FlaggedTransaction], sms_data: List[Dict], mail_data: List[Dict], df: pd.DataFrame) -> List[FlaggedTransaction]:
        results = []
        
        for tx in flagged_tx:
            # Get transaction details
            tx_info = df[df["transaction_id"] == tx.transaction_id].iloc[0]
            sender = tx_info["sender_id"]
            recipient = tx_info["recipient_id"]
            
            # Search for sender/recipient mentions in communications
            relevant_messages = [m for m in mail_data if sender in str(m) or recipient in str(m)]
            relevant_sms = [s for s in sms_data if sender in str(s) or recipient in str(s)]
            
            # If no direct match, look for suspicious keywords generally
            if not relevant_messages:
                relevant_messages = [m for m in mail_data if "urgent" in str(m).lower() or "verify" in str(m).lower()][:3]
            if not relevant_sms:
                relevant_sms = [s for s in sms_data if "link" in str(s).lower() or "click" in str(s).lower()][:3]
            
            prompt = ChatPromptTemplate.from_template("""
            You are an Analyst Agent. You have a flagged transaction and related communications.
            Check if this transaction correlates with any phishing indicators (urgent emails, suspicious links in SMS) involving the participants.
            
            Transaction: {tx_id}
            Sender: {sender}
            Recipient: {recipient}
            Amount: {amount}
            
            Communications involving participants or suspicious keywords:
            Messages: {messages}
            SMS: {sms}
            
            Does this look like a phishing-induced fraud? Provide your updated confidence (0.0 to 1.0) and reason.
            """)
            
            handler = get_langfuse_handler()
            response = self.llm.invoke(
                prompt.format(
                    tx_id=tx.transaction_id, 
                    sender=sender,
                    recipient=recipient,
                    amount=tx_info["amount"],
                    messages=json.dumps(relevant_messages[:5]),
                    sms=json.dumps(relevant_sms[:5])
                ),
                config={"callbacks": [handler]}
            )
            
            # Update transaction based on LLM feedback
            tx.reason += f" | Analyst: {response.content[:150]}..."
            # Simple confidence update based on keywords in response
            if "high" in response.content.lower() or "likely" in response.content.lower():
                tx.confidence = min(1.0, tx.confidence + 0.2)
            
            results.append(tx)
            
        return results
