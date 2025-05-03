import imaplib
import email
import yaml
import logging
import configparser
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import json
import sys
from email.header import decode_header
import html2text

class GmailProcessor:
    """Advanced Gmail email processing class for extracting and saving email data."""
    
    def __init__(self, config_path: str = "gmail_config.ini"):
        """Initialize processor with configuration and logging."""
        self.setup_logging()
        self.load_config(config_path)
        self.mail = None
        self.extracted_data = {
            "Subject": [],
            "From": [],
            "Date": [],
            "Body": [],
            "Attachments": []
        }

    def setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gmail_processor.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str) -> None:
        """Load configuration from INI file."""
        config = configparser.ConfigParser()
        try:
            config.read(config_path)
            self.credentials_file = config.get('Settings', 'credentials_file', fallback='credentials.yml')
            self.imap_url = config.get('Settings', 'imap_url', fallback='imap.gmail.com')
            self.mailbox = config.get('Settings', 'mailbox', fallback='Inbox')
            self.search_key = config.get('Settings', 'search_key', fallback='FROM')
            self.search_value = config.get('Settings', 'search_value', fallback='')
            self.output_dir = config.get('Settings', 'output_dir', fallback='output')
            self.output_format = config.get('Settings', 'output_format', fallback='csv')
            
            # Create output directory if it doesn't exist
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def load_credentials(self) -> tuple:
        """Load credentials from YAML file."""
        try:
            with open(self.credentials_file) as f:
                credentials = yaml.load(f, Loader=yaml.FullLoader)
            return credentials["user"], credentials["password"]
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {str(e)}")
            raise

    def connect_to_gmail(self) -> None:
        """Establish connection to Gmail IMAP server."""
        try:
            user, password = self.load_credentials()
            self.mail = imaplib.IMAP4_SSL(self.imap_url)
            self.mail.login(user, password)
            self.mail.select(self.mailbox)
            self.logger.info(f"Connected to Gmail, selected mailbox: {self.mailbox}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Gmail: {str(e)}")
            raise

    def decode_email_subject(self, subject: str) -> str:
        """Decode email subject with proper handling of encoded headers."""
        try:
            decoded_subject = decode_header(subject)[0][0]
            if isinstance(decoded_subject, bytes):
                charset = decode_header(subject)[0][1] or 'utf-8'
                return decoded_subject.decode(charset)
            return decoded_subject
        except Exception as e:
            self.logger.warning(f"Failed to decode subject: {str(e)}")
            return subject

    def process_email_body(self, part: email.message.Message) -> str:
        """Process email body, handling both plain text and HTML content."""
        try:
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                return part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
            elif content_type == 'text/html':
                html_content = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
                return html2text.html2text(html_content)
            return ""
        except Exception as e:
            self.logger.warning(f"Failed to process email body: {str(e)}")
            return ""

    def save_attachments(self, part: email.message.Message, email_id: str) -> List[str]:
        """Save email attachments and return their filenames."""
        saved_attachments = []
        try:
            if part.get('Content-Disposition') is None:
                return saved_attachments
                
            filename = part.get_filename()
            if filename:
                decoded_filename = decode_header(filename)[0][0]
                if isinstance(decoded_filename, bytes):
                    charset = decode_header(filename)[0][1] or 'utf-8'
                    decoded_filename = decoded_filename.decode(charset)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_filename = f"{timestamp}_{email_id}_{decoded_filename}"
                filepath = Path(self.output_dir) / 'attachments' / safe_filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                with open(filepath, 'wb') as f:
                    f.write(part.get_payload(decode=True))
                saved_attachments.append(str(filepath))
                self.logger.info(f"Saved attachment: {safe_filename}")
                
        except Exception as e:
            self.logger.warning(f"Failed to save attachment: {str(e)}")
        return saved_attachments

    def process_emails(self) -> None:
        """Process emails and extract structured data."""
        try:
            _, data = self.mail.search(None, self.search_key, self.search_value)
            mail_id_list = data[0].split()
            self.logger.info(f"Found {len(mail_id_list)} emails matching criteria")

            for num in mail_id_list:
                try:
                    typ, data = self.mail.fetch(num, '(RFC822)')
                    for response_part in data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            
                            # Extract metadata
                            subject = self.decode_email_subject(msg['subject'] or '')
                            from_addr = msg['from'] or ''
                            date = msg['date'] or ''
                            
                            # Extract body
                            body = ""
                            attachments = []
                            for part in msg.walk():
                                if part.get_content_type() in ['text/plain', 'text/html']:
                                    body += self.process_email_body(part)
                                elif part.get('Content-Disposition'):
                                    attachments.extend(self.save_attachments/part, num.decode()))

                            self.extracted_data["Subject"].append(subject)
                            self.extracted_data["From"].append(from_addr)
                            self.extracted_data["Date"].append(date)
                            self.extracted_data["Body"].append(body)
                            self.extracted_data["Attachments"].append(";".join(attachments))
                            
                except Exception as e:
                    self.logger.warning(f"Error processing email ID {num}: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"Error processing emails: {str(e)}")
            raise

    def save_output(self) -> None:
        """Save extracted data to specified format."""
        try:
            df = pd.DataFrame(self.extracted_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.output_dir) / f"extracted_emails_{timestamp}"

            if self.output_format.lower() == 'csv':
                output_file = output_path.with_suffix(".csv")
                df.to_csv(output_file, index=False)
            elif self.output_format.lower() == 'json':
                output_file = output_path.with_suffix(".json")
                df.to_json(output_file, orient='records', lines=True)
            else:
                raise ValueError(f"Unsupported output format: {self.output_format}")

            self.logger.info(f"Data saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save output: {str(e)}")
            raise

    def run(self) -> None:
        """Execute the email processing pipeline."""
        try:
            self.connect_to_gmail()
            self.process_emails()
            if self.extracted_data["Subject"]:  # Only save if data was extracted
                self.save_output()
            else:
                self.logger.warning("No data extracted, skipping save operation")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        finally:
            if self.mail:
                self.mail.logout()
                self.logger.info("Disconnected from Gmail")


if __name__ == "__main__":
    # Create default configuration file if not exists
    config_content = """
    [Settings]
    credentials_file = credentials.yml
    imap_url = imap.gmail.com
    mailbox = Inbox
    search_key = FROM
    search_value = bnsreenu@hotmail.com
    output_dir = output
    output_format = csv
    """
    
    config_path = "gmail_config.ini"
    if not Path(config_path).exists():
        with open(config_path, 'w') as f:
            f.write(config_content.strip())

    processor = GmailProcessor(config_path)
    processor.run()
