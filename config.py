import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(BASE_DIR, "Sample_HR_Policy_Document.pdf")
LOGO_PATH = os.path.join(BASE_DIR, "nexus_iq_logo.png")
ANALYTICS_PATH = os.path.join(BASE_DIR, "analytics.csv")

USERS = {
    "employee": {"password": "employee123", "role": "Employee"},
    "hr": {"password": "hr123", "role": "HR"}
}
