from enum import Enum


class StatusCodes(Enum):
    SUCCESS = "Success"
    ERROR = "Error"
    RUNNING_ANALYSIS = "Running analysis"
    RUNNING_GENERATION = "Running generation"
    RECEIVED = "Received"
    IDLING = "Idling"
