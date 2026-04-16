import os
import pandas as pd
import json
import re
from typing import List
from .schemas import Transaction, SMS, Mail, User, Location, DatasetPackage
from langfuse import observe

class DataAgent:
    def __init__(self, dataset_path: str):
        self.path = dataset_path

    def _parse_sms(self, content: str) -> SMS:
        sender = re.search(r"From:\s*(.*)", content)
        receiver = re.search(r"To:\s*(.*)", content)
        timestamp = re.search(r"Date:\s*(.*)", content)
        message = re.search(r"Message:\s*(.*)", content, re.DOTALL)
        
        return SMS(
            sender=sender.group(1).strip() if sender else "Unknown",
            receiver=receiver.group(1).strip() if receiver else "Unknown",
            timestamp=timestamp.group(1).strip() if timestamp else "Unknown",
            text=message.group(1).strip() if message else content
        )

    def _parse_mail(self, content: str) -> Mail:
        sender = re.search(r"From:\s*(.*)", content)
        receiver = re.search(r"To:\s*(.*)", content)
        subject = re.search(r"Subject:\s*(.*)", content)
        timestamp = re.search(r"Date:\s*(.*)", content)
        
        # Body starts after double newline or after the last header
        parts = re.split(r"\n\s*\n", content, maxsplit=1)
        body = parts[1] if len(parts) > 1 else content
        # Basic HTML strip
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()

        return Mail(
            sender=sender.group(1).strip() if sender else "Unknown",
            receiver=receiver.group(1).strip() if receiver else "Unknown",
            subject=subject.group(1).strip() if subject else "No Subject",
            timestamp=timestamp.group(1).strip() if timestamp else "Unknown",
            content=body
        )

    @observe()
    def load_and_structure(self) -> DatasetPackage:
        # Transactions
        tx_df = pd.read_csv(os.path.join(self.path, "transactions.csv"))
        tx_df = tx_df.replace({float('nan'): None})
        transactions = [Transaction(**row) for row in tx_df.to_dict(orient="records")]

        # SMS
        with open(os.path.join(self.path, "sms.json"), "r") as f:
            sms_data = json.load(f)
            sms = [self._parse_sms(item["sms"]) for item in sms_data]

        # Mails
        with open(os.path.join(self.path, "mails.json"), "r") as f:
            mail_data = json.load(f)
            mails = [self._parse_mail(item["mail"]) for item in mail_data]

        # Users
        with open(os.path.join(self.path, "users.json"), "r") as f:
            users_data = json.load(f)
            users = [User(**item) for item in users_data]

        # Locations
        with open(os.path.join(self.path, "locations.json"), "r") as f:
            loc_data = json.load(f)
            locations = [Location(**item) for item in loc_data]

        # Audio files detection
        audio_dir = os.path.join(self.path, "audio")
        audio_files = []
        if os.path.exists(audio_dir):
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]

        return DatasetPackage(
            transactions=transactions,
            sms=sms,
            mails=mails,
            users=users,
            locations=locations,
            audio_files=audio_files
        )
