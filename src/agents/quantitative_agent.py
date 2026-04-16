import pandas as pd
from typing import List
from .base import get_llm, get_langfuse_handler
from .schemas import DatasetPackage, QuantitativeSignal
from .tools import get_impossible_travel_report, get_financial_risk_report, get_spatial_profile_report
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langfuse import observe

class QuantitativeAgent:
    def __init__(self):
        self.llm = get_llm()
        self.parser = PydanticOutputParser(pydantic_object=QuantitativeSignal)

    def _triage_transactions(self, data: DatasetPackage) -> List[str]:
        """Use pandas to find suspicious candidates for LLM analysis."""
        df = pd.DataFrame([tx.model_dump() for tx in data.transactions])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 1. High Velocity
        df = df.sort_values(['sender_id', 'timestamp'])
        df['time_diff'] = df.groupby('sender_id')['timestamp'].diff().dt.total_seconds() / 60
        velocity_mask = (df['time_diff'] < 60) & (df['time_diff'] > 0)
        
        # 2. Large Amount Outliers
        user_median = df.groupby('sender_id')['amount'].transform('median')
        amount_mask = df['amount'] > (user_median * 3)
        
        # 3. Late Night
        hour = df['timestamp'].dt.hour
        night_mask = (hour >= 1) & (hour <= 5)
        
        # 4. Global Top amounts
        top_mask = df['amount'] > df['amount'].quantile(0.95)

        suspicious_ids = df[velocity_mask | amount_mask | night_mask | top_mask]['transaction_id'].unique()
        return list(suspicious_ids)[:50]

    @observe()
    def analyze(self, data: DatasetPackage, session_id: str = None) -> List[QuantitativeSignal]:
        candidate_ids = self._triage_transactions(data)
        candidates = [tx for tx in data.transactions if tx.transaction_id in candidate_ids]
        
        signals = []
        for tx in candidates:
            user = next((u for u in data.users if u.iban == tx.sender_iban or u.iban == tx.sender_id), None)
            
            # Reports
            financial_report = get_financial_risk_report(tx, user)
            travel_report = get_impossible_travel_report(tx, data.transactions, data.locations)
            spatial_report = get_spatial_profile_report(tx, user, data.locations)
            
            prompt = ChatPromptTemplate.from_template("""
            Role: Advanced Quantitative Fraud Analyst.
            
            Transaction: {tx_json}
            
            --- FINANCIAL RISK REPORT ---
            {fin_report}
            
            --- TRAVEL RISK REPORT ---
            {travel_report}

            --- SPATIAL PROFILING REPORT (Habits) ---
            {spatial_report}
            
            Analyze these reports. 
            - Focus on 'SPATIAL PROFILING': Even if a journey is possible, is it habit-consistent?
            - Impossible travel is a critical indicator.
            - Extreme financial drain relative to salary is high risk.
            
            Return a QuantitativeSignal.
            {format_instructions}
            """)
            
            handler = get_langfuse_handler()
            if session_id: handler.session_id = session_id
                
            response = self.llm.invoke(
                prompt.format(
                    tx_json=tx.model_dump_json(),
                    fin_report=financial_report,
                    travel_report=travel_report,
                    spatial_report=spatial_report,
                    format_instructions=self.parser.get_format_instructions()
                ),
                config={"callbacks": [handler]}
            )
            try:
                signals.append(self.parser.parse(response.content))
            except: continue
        return signals
