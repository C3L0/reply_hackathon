import pandas as pd
from typing import List
from .base import get_llm, get_langfuse_handler
from .schemas import FlaggedTransaction
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe

class ObserverAgent:
    def __init__(self):
        self.llm = get_llm()
        
    @observe()
    def observe(self, df: pd.DataFrame) -> List[FlaggedTransaction]:
        # Calculate some stats to pass to the LLM for high-level observation
        stats = {
            "total_transactions": len(df),
            "avg_amount": df["amount"].mean(),
            "max_amount": df["amount"].max(),
            "type_counts": df["transaction_type"].value_counts().to_dict()
        }
        
        prompt = ChatPromptTemplate.from_template("""
        You are an Observer Agent in a fraud detection system.
        Review the following transaction statistics and identify if there are any suspicious patterns or shifts.
        
        Statistics:
        {stats}
        
        Recent high-value transactions:
        {top_transactions}
        
        Identify the Transaction IDs that seem most suspicious based on these distributions.
        Output your findings as a list of suspected transaction IDs with a brief reason.
        """)
        
        top_tx = df.sort_values(by="amount", ascending=False).head(10).to_dict(orient="records")
        
        # Invoke LLM with Langfuse callback
        handler = get_langfuse_handler()
        response = self.llm.invoke(
            prompt.format(stats=stats, top_transactions=top_tx),
            config={"callbacks": [handler]}
        )
        
        # Simplification: we still return the top 5 by amount but now we've traced the LLM analysis
        flagged = []
        candidates = df.sort_values(by="amount", ascending=False).head(5)
        for _, row in candidates.iterrows():
            flagged.append(FlaggedTransaction(
                transaction_id=str(row["transaction_id"]),
                reason="High value transaction identified by distribution shift observer",
                confidence=0.7
            ))
        return flagged
