
import pandas as pd
import re

# PATTERN = re.compile(r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\w+)\s+((\w+)(?:\((\w+)\))?(?:\[(\d+)\])?):\s*(.*)')
path = "../data/Linux.log"
date_pattern = r"(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})"
process_pattern = r"combo\s+(\w+)(?:\[\d+\])?"
process_id_pattern = r"(\w+)\[(\d+)\]:"
message_pattern = r": (.*)"

class LogEntry:
    def __init__(self, path):
        self.path = path
        self.liste_log = []
        self.erreur = ""
        self.bad_parse = []

    def log_parser(self):
        with open(path, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                date = re.search(date_pattern, line)
                process = re.search(process_pattern, line)
                process_id = re.search(process_id_pattern, line)
                message = re.search(message_pattern, line)

                # Extraire les valeurs ou None si non trouvées
                date_val = date.group(1) if date else None
                process_val = process.group(1) if process else None
                process_id_val = process_id.group(2) if process_id else None
                message_val = message.group(1) if message else None

                self.liste_log.append([date_val, process_val, process_id_val, message_val])

    def liste_to_dataframe(self):
        columns = ["Date", "Process", "Process ID", "Message"]
        self.df = pd.DataFrame(self.liste_log, columns=columns)

    def date_format(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%b %d %H:%M:%S', errors='coerce')

    def process(self):
        self.log_parser()
        self.liste_to_dataframe()
        self.date_format()
        return self.df, self.bad_parse


df, erreur = LogEntry(path).process()
