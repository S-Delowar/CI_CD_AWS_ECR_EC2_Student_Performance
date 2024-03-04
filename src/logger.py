import logging
import os
from datetime import datetime

LOG_FILE = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'

logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

# inside the log_path, we will print our log. thats why, need a file where the logs with generated
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

#Set the basic configurations 
logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= "[ %(asctime)s] file:%(filename)s line no:%(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO
)

