from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class SessionInformation(BaseModel):
    """A model with information about a single user session"""

    session_id: int
    site1: int
    time1: datetime
    site2: Optional[int]
    time2: Optional[datetime]
    site3: Optional[int]
    time3: Optional[datetime]
    site4: Optional[int]
    time4: Optional[datetime]
    site5: Optional[int]
    time5: Optional[datetime]
    site6: Optional[int]
    time6: Optional[datetime]
    site7: Optional[int]
    time7: Optional[datetime]
    site8: Optional[int]
    time8: Optional[datetime]
    site9: Optional[int]
    time9: Optional[datetime]
    site10: Optional[int]
    time10: Optional[datetime]
