import os

from langfuse import observe

from src.agents.base import generate_session_id, langfuse_client
from src.agents.data_agent import DataAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.nlp_agent import NLPAgent
from src.agents.quantitative_agent import QuantitativeAgent


@observe()
def run_fraud_detection(dataset_path: str):
    # 1. Setup Session ID
    session_id = generate_session_id()

    # We use manual session linking in handlers since langfuse.context isn't working
    print(f"\n--- Starting Fraud Detection ---")
    print(f"Dataset: {dataset_path}")
    print(f"SESSION_ID: {session_id}")
    print(f"---" * 10)

    # 2. Data Ingestion & Structuring
    data_agent = DataAgent(dataset_path)
    print(f"Loading data from {dataset_path}...")
    dataset = data_agent.load_and_structure()

    # 3. Quantitative Analysis
    quant_agent = QuantitativeAgent()
    print(f"Running Quantitative Analysis...")
    # Pass session_id to explicitly link it in CallbackHandler
    quant_signals = quant_agent.analyze(dataset, session_id=session_id)

    # 4. NLP & Phishing Analysis
    nlp_agent = NLPAgent()
    print(f"Running NLP Analysis...")
    nlp_signals = nlp_agent.analyze(dataset, session_id=session_id)

    # 5. Final Decision
    decision_agent = DecisionAgent()
    print(f"Finalizing decisions and saving to detected_fraud.txt...")
    fraudulent_ids = decision_agent.finalize(
        quant_signals,
        nlp_signals,
        dataset,
        output_path="detected_fraud.txt",
        session_id=session_id,
    )

    print(f"\nDetection Complete.")
    print(f"Flagged {len(fraudulent_ids)} suspicious transactions.")
    print(f"IMPORTANT: Use this Session ID for your submission: {session_id}")

    # Force flush to ensure traces are uploaded
    langfuse_client.flush()


if __name__ == "__main__":
    # Configure your dataset here (1, 2, or 3)
    TARGET_DATASET = "data/dataset3_train"

    if not os.path.exists(TARGET_DATASET):
        print(f"Error: Dataset path {TARGET_DATASET} not found.")
    else:
        run_fraud_detection(TARGET_DATASET)
