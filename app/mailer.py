from __future__ import annotations
import smtplib
from email.message import EmailMessage
from app.config import settings


def send_verification(email: str, token: str):
    url=f"{settings.public_url.rstrip('/')}/?verify={token}"
    subject="Verify your Eidomira account"
    body=f"Verify your email to activate your 7-day Eidomira trial:\n\n{url}\n\nThis link expires in {settings.email_token_ttl//3600} hours. If you did not register, ignore this message."
    if not settings.smtp_host:
        print(f"EIDOMIRA DEVELOPMENT VERIFICATION for {email}: {url}")
        return
    message=EmailMessage(); message["Subject"]=subject; message["From"]=settings.smtp_from; message["To"]=email; message.set_content(body)
    with smtplib.SMTP(settings.smtp_host,settings.smtp_port,timeout=15) as smtp:
        if settings.smtp_starttls: smtp.starttls()
        if settings.smtp_username: smtp.login(settings.smtp_username,settings.smtp_password)
        smtp.send_message(message)
