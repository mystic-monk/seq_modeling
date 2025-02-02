# tests/test_logger.py
import os
import logging
import pytest
import re
from utils.logger import setup_logger

@pytest.fixture(scope="module")
def setup_test_logger():
    """Fixture to set up the logger and ensure cleanup."""
    log_dir = "test_logs"
    log_file = "test_logfile.log"
    
    # Setup the logger
    logger = setup_logger(log_dir, log_file, log_level=logging.DEBUG)
    
    # Yield the logger to the test
    yield logger

    # Cleanup: Remove the log files after the test
    log_file_path = os.path.join(log_dir, log_file)
    
    # Close logger handlers
    for handler in logger.handlers:
        handler.close()
    
    # Now safely remove the log files
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    if os.path.exists(log_dir):
        os.rmdir(log_dir)

# Test to verify that the log file is created
def test_logger_creates_log_file(setup_test_logger):
    log_dir = "test_logs"
    log_file = "test_logfile.log"
    log_file_path = os.path.join(log_dir, log_file)
    
    # Log a test message
    logger = setup_test_logger
    logger.debug("This is a test debug message.")
    
    # Check if the log file is created
    assert os.path.exists(log_file_path)

# Test to verify that log messages are written to the file
def test_logger_writes_to_file(setup_test_logger):
    log_dir = "test_logs"
    log_file = "test_logfile.log"
    log_file_path = os.path.join(log_dir, log_file)
    
    # Log a test message
    logger = setup_test_logger
    logger.info("This is an info message.")
    
    # Read the log file and check for the expected message
    with open(log_file_path, "r", encoding="utf-8") as file:
        logs = file.read()
        assert "This is an info message." in logs

# Test to verify that log messages are output to the console
def test_logger_output_to_console(setup_test_logger, capfd):
    # Log a test message
    logger = setup_test_logger
    logger.warning("This is a warning message.")
    
    # Capture the console output
    captured = capfd.readouterr()

    # Use regex to find the log message in the output
    # We expect the log message to be somewhere in the captured output
    match = re.search(r"WARNING.*This is a warning message.", captured.out)
    
    # Assert that the message is found
    assert match is not None, f"Log message not found in console output. Captured: {captured.out}"

# Test to verify log level configuration
def test_logger_log_level(setup_test_logger, capfd):
    # Setup logger with DEBUG level
    logger = setup_test_logger
    logger.setLevel(logging.DEBUG)
    
    # Log messages with different levels
    logger.debug("Debug message")
    logger.info("Info message")
    
    # Capture the console output
    captured = capfd.readouterr()
    
    # Assert that both debug and info messages are present in the console output
    assert "Debug message" in captured.out
    assert "Info message" in captured.out

    # Verify log file contents
    log_dir = "test_logs"
    log_file = "test_logfile.log"
    log_file_path = os.path.join(log_dir, log_file)
    with open(log_file_path, "r", encoding="utf-8") as file:
        logs = file.read()
        assert "Debug message" in logs
        assert "Info message" in logs
