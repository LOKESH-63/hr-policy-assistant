import pandas as pd
from datetime import datetime
from config import ANALYTICS_PATH


def init_analytics():
    if not pd.io.common.file_exists(ANALYTICS_PATH):
        pd.DataFrame(
            columns=["timestamp", "user", "role", "question", "answered"]
        ).to_csv(ANALYTICS_PATH, index=False)


def log_query(user, role, question, answered):
    df = pd.read_csv(ANALYTICS_PATH)
    df.loc[len(df)] = [
        datetime.now(),
        user,
        role,
        question,
        answered
    ]
    df.to_csv(ANALYTICS_PATH, index=False)
