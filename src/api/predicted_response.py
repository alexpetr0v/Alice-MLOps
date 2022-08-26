from pydantic import BaseModel


class PredictedResponse(BaseModel):
    """Predicted response for the session"""

    session_id: int
    class_0_proba: float
    class_1_proba: float
