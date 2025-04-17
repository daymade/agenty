# test_log.py
import logging
import sys

print(">>> test_log.py: Script started", file=sys.stderr) # Direct print to stderr just in case logging fails

try:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)
    logger = logging.getLogger(__name__)
    print(">>> test_log.py: Logging configured", file=sys.stderr)
    logger.info(">>> test_log.py: This is an INFO log message.")
    print(">>> test_log.py: Log message sent", file=sys.stderr)
except Exception as e:
    print(f">>> test_log.py: ERROR during logging setup/use: {e}", file=sys.stderr)

print(">>> test_log.py: Script finished", file=sys.stderr)
