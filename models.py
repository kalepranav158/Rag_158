from pydantic import BaseModel, Field
from typing import Literal, List

# -----------------------------
#       PYDANTIC MODELS
# -----------------------------

class RetriveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer, else false."
    )

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True if the document helps to answer the question, else False."
    )

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "not_supported"] = Field(
        ...,
        description="fully_supported if the answer is fully supported by the context, partially_supported if the answer is partially supported by the context, not_supported if the answer is not supported by the context."
    )
    evidence: List[str] = Field(
        ...,
        description="List of evidence from the context that supports the answer."
    )

class IsUSEDecision(BaseModel):
    issue: Literal["useful", "not_useful"] = Field(
        ...,
        description="useful if the answer is useful for the user, not_useful if the answer is not useful for the user."
    )
    reason: str = Field(
        ...,
        description="A short reason explaining why the answer is useful or not useful."
    )