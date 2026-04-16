from typing import List
from .base import get_llm, get_langfuse_handler
from .schemas import FlaggedTransaction, FraudReport
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langfuse import observe

class ExecutorAgent:
    def __init__(self):
        self.llm = get_llm()
        self.parser = PydanticOutputParser(pydantic_object=FraudReport)
        
    @observe()
    def finalize(self, analyzed_tx: List[FlaggedTransaction]) -> FraudReport:
        prompt = ChatPromptTemplate.from_template("""
        You are the Executor Agent. You make the final decision on which transactions are fraudulent.
        Review the following analyzed transactions and their phishing correlation.
        
        Analyzed Transactions:
        {data}
        
        Decide which ones should be officially flagged as fraud.
        Return only the final list of fraudulent Transaction IDs and a summary justification.
        
        {format_instructions}
        """)
        
        input_data = "\n".join([f"ID: {t.transaction_id}, Confidence: {t.confidence}, Reason: {t.reason}" for t in analyzed_tx])
        
        handler = get_langfuse_handler()
        response = self.llm.invoke(
            prompt.format(
                data=input_data,
                format_instructions=self.parser.get_format_instructions()
            ),
            config={"callbacks": [handler]}
        )
        
        return self.parser.parse(response.content)
