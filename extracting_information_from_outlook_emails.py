import win32com.client
import pandas as pd
import re
import logging
import configparser
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import sys

class EmailProcessor:
    """Advanced email processing class for extracting structured data from Outlook emails."""
    
    def __init__(self, config_path: str = "config.ini"):
        """Initialize processor with configuration and logging."""
        self.setup_logging()
        self.load_config(config_path)
        self.outlook = None
        self.inbox = None
        self.extracted_data = {
            "Name": [],
            "Company": [],
            "Email": [],
            "Message": [],
            "ReceivedTime": []
        }

    def setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('email_processor.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str) -> None:
        """Load configuration from INI file."""
        config = configparser.ConfigParser()
        try:
            config.read(config_path)
            self.folder_index = config.getint('Settings', 'folder_index', fallback=6)
            self.subject_filter = config.get('Settings', 'subject_filter', fallback="From Sreeni")
            self.output_dir = config.get('Settings', 'output_dir', fallback="output")
            self.output_format = config.get('Settings', 'output_format', fallback="excel")
            
            # Create output directory if it doesn't exist
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def connect_to_outlook(self) -> None:
        """Establish connection to Outlook."""
        try:
            self.outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
            self.inbox = self.outlook.GetDefaultFolder(self.folder_index)
            self.logger.info(f"Connected to Outlook, accessing folder index: {self.folder_index}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Outlook: {str(e)}")
            raise

    def extract_field(self, text: str, field: str, next_field: Optional[str] = None) -> str:
        """Extract specific field using regex for robust parsing."""
        try:
            # Define regex patterns for different fields
            patterns = {
                "Name": r"Name:\s*([^\n]+?)(?=\n(?:Company:|$))",
                "Company": r"Company:\s*([^\n]+?)(?=\n(?:Email:|$))",
                "Email": r"Email:\s*([^\n]+?)(?=\n(?:Message:|$))",
                "Message": r"Message:\s*([\s\S]+)"
            }
            
            pattern = patterns.get(field, "")
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(1).strip() if match else ""
        except Exception as e:
            self.logger.warning(f"Failed to extract {field}: {str(e)}")
            return ""

    def process_emails(self) -> None:
        """Process emails and extract structured data."""
        try:
            messages = self.inbox.Items
            message_count = 0

            for message in messages:
                try:
                    if message.Subject == self.subject_filter:
                        message_count += 1
                        body = message.Body

                        # Extract fields
                        self.extracted_data["Name"].append(self.extract_field(body, "Name"))
                        self.extracted_data["Company"].append(self.extract_field(body, "Company"))
                        self.extracted_data["Email"].append(self.extract_field(body, "Email"))
                        self.extracted_data["Message"].append(self.extract_field(body, "Message"))
                        self.extracted_data["ReceivedTime"].append(
                            message.ReceivedTime.strftime("%Y-%m-%d %H:%M:%S")
                        )
                except Exception as e:
                    self.logger.warning(f"Error processing email: {str(e)}")
                    continue

            self.logger.info(f"Processed {message_count} emails with subject '{self.subject_filter}'")
            
        except Exception as e:
            self.logger.error(f"Error processing emails: {str(e)}")
            raise

    def save_output(self) -> None:
        """Save extracted data to specified format."""
        try:
            df = pd.DataFrame(self.extracted_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.output_dir) / f"extracted_info_{timestamp}"

            if self.output_format.lower() == "excel":
                output_file = output_path.with_suffix(".xlsx")
                df.to_excel(output_file, index=False)
            elif self.output_format.lower() == "csv":
                output_file = output_path.with_suffix(".csv")
                df.to_csv(output_file, index=False)
            else:
                raise ValueError(f"Unsupported output format: {self.output_format}")

            self.logger.info(f"Data saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save output: {str(e)}")
            raise

    def run(self) -> None:
        """Execute the email processing pipeline."""
        try:
            self.connect_to_outlook()
            self.process_emails()
            if self.extracted_data["Name"]:  # Only save if data was extracted
                self.save_output()
            else:
                self.logger.warning("No data extracted, skipping save operation")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Create default configuration file if not exists
    config_content = """
    [Settings]
    folder_index = 6
    subject_filter = From Sreeni
    output_dir = output
    output_format = excel
    """
    
    config_path = "config.ini"
    if not Path(config_path).exists():
        with open(config_path, 'w') as f:
            f.write(config_content.strip())

    processor = EmailProcessor(config_path)
    processor.run()
