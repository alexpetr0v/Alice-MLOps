import logging
import os

# set loglevel from environment variables
LOGLEVEL = os.environ.get("LOGLEVEL")
if LOGLEVEL:
    logging.basicConfig(level=LOGLEVEL.upper())
