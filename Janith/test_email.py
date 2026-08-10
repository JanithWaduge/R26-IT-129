import smtplib
from email.mime.text import MIMEText

SMTP_USER = "hansamanajanith11@gmail.com"
SMTP_PASS = "htncnubahhwwbmut"  # no spaces
TO = "kisaldulmin.002@gmail.com"

msg = MIMEText("This is a test email from the SLSL project.")
msg["Subject"] = "SLSL Test Email"
msg["From"] = SMTP_USER
msg["To"] = TO

try:
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
    print("✅ SUCCESS — email sent!")
except Exception as e:
    print(f"❌ FAILED: {e}")