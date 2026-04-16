from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Data Schemas ---

class Transaction(BaseModel):
    transaction_id: str
    sender_id: str
    recipient_id: Optional[Any] = None
    transaction_type: Optional[str] = None
    amount: float
    location: Optional[Any] = None
    payment_method: Optional[Any] = None
    sender_iban: Optional[Any] = None
    recipient_iban: Optional[Any] = None
    balance_after: float
    description: Optional[Any] = None
    timestamp: str

class SMS(BaseModel):
    sender: str
    receiver: str
    timestamp: str
    text: str

class Mail(BaseModel):
    sender: str
    receiver: str
    timestamp: str
    subject: str
    content: str

class User(BaseModel):
    first_name: str
    last_name: str
    job: str
    iban: str
    residence: Dict[str, Any]
    salary: Optional[float] = None
    birth_year: Optional[int] = None
    description: Optional[str] = None

class Location(BaseModel):
    biotag: str
    timestamp: str
    lat: float
    lng: float
    city: Optional[str] = None

class DatasetPackage(BaseModel):
    transactions: List[Transaction]
    sms: List[SMS]
    mails: List[Mail]
    users: List[User]
    locations: List[Location]
    audio_files: List[str] = []

# --- Agent Signal Schemas ---

class QuantitativeSignal(BaseModel):
    transaction_id: str
    frequency_risk: float = Field(description="Score for unusual frequency of transactions")
    temporal_risk: float = Field(description="Score for unusual timing")
    geo_risk: float = Field(description="Score for impossible geographic jumps")
    demographic_risk: float = Field(description="Score for inconsistencies between user profile (age/job) and transaction")
    amount_weight: float = Field(description="Weighting factor based on transaction amount")
    quantitative_score: float = Field(description="Aggregate quantitative risk score (0-1)")
    reasoning: Optional[str] = Field(None, description="Brief explanation of the quantitative risk")

class NLPSignal(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Optional: ID of a specific transaction if directly linked")
    target_name: Optional[str] = Field(None, description="Name of the person targeted in the communication")
    target_email: Optional[str] = Field(None, description="Email of the person targeted")
    target_phone: Optional[str] = Field(None, description="Phone number of the person targeted")
    phishing_vector: float = Field(description="Score for phishing indicators in communications")
    urgency_score: float = Field(description="Score for detected sentiment of urgency")
    ioc_list: List[str] = Field(default_factory=list, description="List of Indicators of Compromise found (URLs, suspicious phones, etc.)")
    audio_analysis: Optional[str] = Field(None, description="Summary of audio transcription analysis if applicable")
    nlp_score: float = Field(description="Aggregate NLP risk score (0-1)")
    reasoning: Optional[str] = Field(None, description="Brief explanation of the NLP risk")

class FinalDecision(BaseModel):
    transaction_id: str
    is_fraud: bool
    total_risk_score: float
    justification: str
