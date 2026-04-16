import math
from datetime import datetime
from typing import Optional, List, Dict
from .schemas import Transaction, User, Location
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth."""
    R = 6371  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_impossible_travel_report(tx: Transaction, all_tx: List[Transaction], locations: List[Location]) -> str:
    """Checks for impossible travel between current transaction and previous events."""
    # This is a helper for the LLM, not a deterministic rule
    
    # 1. Try to find the previous transaction for this user
    user_tx = sorted([t for t in all_tx if t.sender_id == tx.sender_id and t.timestamp < tx.timestamp], 
                     key=lambda x: x.timestamp)
    
    # 2. Extract current coordinates if possible (some tx might have them in 'location' string or we look up city)
    # Since location data in CSV is often city names, we'd ideally have a city->coord map.
    # For now, let's use the GPS locations as the ground truth for "where the user was".
    
    user_prefix = tx.sender_id[:4].upper() # Heuristic for biotag matching
    user_gps = sorted([l for l in locations if l.biotag.startswith(user_prefix) and l.timestamp < tx.timestamp],
                      key=lambda x: x.timestamp)
    
    if not user_gps:
        return "No historical GPS data found for this user."

    last_gps = user_gps[-1]
    
    # Simple check: if current tx location is a city, and last GPS was elsewhere
    # (In a real implementation we would geocode tx.location)
    # Here we return the last known position to let the LLM judge.
    
    return f"Last known GPS location: {last_gps.city} ({last_gps.lat}, {last_gps.lng}) at {last_gps.timestamp}. " \
           f"Transaction location: {tx.location}. Time elapsed: {tx.timestamp}."

def get_financial_risk_report(tx: Transaction, user: Optional[User]) -> str:
    """Analyzes amount relative to salary and balance."""
    if not user:
        return "User profile not found."
    
    monthly_salary = user.salary / 12 if user.salary else 0
    balance_before = tx.balance_after + tx.amount
    
    ratio_salary = (tx.amount / monthly_salary) * 100 if monthly_salary > 0 else 0
    ratio_balance = (tx.amount / balance_before) * 100 if balance_before > 0 else 0
    
    report = f"Transaction amount: {tx.amount}. User monthly salary: {monthly_salary:.2f}. "
    report += f"This transaction is {ratio_salary:.1f}% of their monthly income. "
    report += f"Balance before transaction: {balance_before:.2f} (Transaction consumes {ratio_balance:.1f}% of total balance)."
    
    if ratio_salary > 200:
        report += " | WARNING: Extremely high amount relative to income."
    if ratio_balance > 80:
        report += " | WARNING: Transaction nearly empties the account."
        
    return report

def get_spatial_profile_report(tx: Transaction, user: Optional[User], locations: List[Location]) -> str:
    """Analyse si la transaction est dans la 'sphère de vie' habituelle de l'utilisateur."""
    if not user:
        return "Profil utilisateur inconnu."
        
    user_prefix = tx.sender_id[:4].upper()
    user_locations = [l for l in locations if l.biotag.startswith(user_prefix)]
    
    if not user_locations:
        return "Aucun historique GPS pour ce profil."
        
    # Extraire les villes fréquentées
    frequent_cities = set([l.city for l in user_locations if l.city])
    
    # Vérifier si la localisation de la transaction est connue
    tx_location = str(tx.location or "")
    is_known_zone = any(city.lower() in tx_location.lower() for city in frequent_cities if city)
    
    report = f"Villes fréquentées (GPS) : {', '.join(list(frequent_cities)[:5])}. "
    if not is_known_zone and tx.transaction_type == "in-person payment":
        report += f" | CRITICAL: Transaction hors de la zone de vie habituelle ({tx_location})."
    elif not is_known_zone:
        report += f" | WARNING: Localisation inhabituelle pour ce profil ({tx_location})."
    else:
        report += " | Localisation cohérente avec les habitudes."
        
    return report
