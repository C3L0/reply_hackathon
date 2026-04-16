# Plan: Comprehensive Improvements for Fraud Detection

## 1. Robust Data Parsing (`DataAgent`)
The current parsing is fragile. I will:
- Implement `re` (regex) based parsing for SMS and Emails.
- Extract `target_name`, `target_contact` (phone/email), and `timestamp` reliably.
- Handle HTML content in emails more gracefully (strip tags or focus on text).

## 2. Advanced Candidate Selection (`QuantitativeAgent`)
Instead of just picking the top 35 transactions by amount, I will:
- Use `pandas` to identify:
    - **Amount Outliers**: Transactions significantly higher than the user's average.
    - **High Velocity**: Multiple transactions in a short time window.
    - **Anomalous Hours**: Transactions between 1 AM and 5 AM.
    - **Impossible Travel**: Basic speed check between GPS/Transactions.
- Pass these "Triggered" transactions to the LLM for deep analysis.

## 3. Improved Signal Linking (`DecisionAgent`)
- Update `NLPSignal` schema to include `target_user_info`.
- In `DecisionAgent`, map `target_user_info` to `sender_id` or `iban`.
- If a user is a confirmed phishing victim, flag their transactions with a higher risk multiplier.

## 4. Schema Updates (`schemas.py`)
- Modify `NLPSignal` to make `transaction_id` optional.
- Add fields for identified targets in communications.

## 5. Implementation Steps
1.  **Modify `schemas.py`** to support improved signal metadata.
2.  **Update `DataAgent`** with regex-based parsing.
3.  **Update `QuantitativeAgent`** with the new triage/filtering logic.
4.  **Update `NLPAgent`** to focus on target identification.
5.  **Update `DecisionAgent`** to link these improved signals.
6.  **Verify** by running `main.py` on `dataset1_train`.
