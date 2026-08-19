"""Pydantic schemas for llama.cpp / OpenAI-compatible structured JSON."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SafetyNotes(BaseModel):
    mode: str = Field(default="", description="safe or smart")
    notes: str = Field(default="", description="short free-text note")


class OverlapSplit(BaseModel):
    from_ts: str = Field(description="source timestamp MM:SS")
    from_spk: int = Field(description="source speaker id")
    keep: str = Field(default="", description="text that stays on the source line")
    foreign: str = Field(default="", description="other-speaker exact substring to move")
    foreign_to_ts: str = Field(default="", description="destination timestamp of the other speaker")
    foreign_to_spk: int = Field(default=0, description="other speaker id; 0 if unused")
    stem: str = Field(default="", description="leftover of the same speaker")
    stem_to_ts: str = Field(default="", description="next same-speaker timestamp")
    stem_to_spk: int = Field(default=0, description="same speaker id; 0 if unused")


class SpellingReplacement(BaseModel):
    src: str = Field(default="", description="exact original substring")
    dst: str = Field(default="", description="corrected spelling")


class SmartRefineResponse(BaseModel):
    splits: List[OverlapSplit] = Field(
        max_length=6,
        description="only real overlap/echo/stem moves; empty if none",
    )
    replacements: List[SpellingReplacement] = Field(
        max_length=8,
        description="ASR spelling fixes; empty if none",
    )
    safety: SafetyNotes


class SafeRefineResponse(BaseModel):
    replacements: List[SpellingReplacement] = Field(
        max_length=8,
        description="ASR spelling/punctuation token fixes; empty if none",
    )
    safety: SafetyNotes


class RolesGuess(BaseModel):
    agent: Optional[int] = None
    client: Optional[int] = None
    manager: Optional[int] = None


class Participants(BaseModel):
    speakers: List[int] = Field(default_factory=list)
    roles_guess: RolesGuess = Field(default_factory=RolesGuess)


class TimelineEvent(BaseModel):
    t_hint: str = Field(default="", description="start|mid|end")
    event: str = Field(default="")


class Entities(BaseModel):
    companies: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    phones: List[str] = Field(default_factory=list)
    inn: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    amounts: List[str] = Field(default_factory=list)
    addresses: List[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    who: str = Field(default="")
    action: str = Field(default="")
    deadline: Optional[str] = None


class IssueItem(BaseModel):
    issue: str = Field(default="")
    evidence: str = Field(default="")
    severity: str = Field(default="low")


class QualityNotes(BaseModel):
    has_transfer: bool = False
    transfer_reason: Optional[str] = None
    asr_uncertainty: Optional[str] = None


class CallSummaryResponse(BaseModel):
    call_id: str = ""
    language: str = "ru"
    participants: Participants = Field(default_factory=Participants)
    intent: str = ""
    topics: List[str] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    entities: Entities = Field(default_factory=Entities)
    actions: List[ActionItem] = Field(default_factory=list)
    issues_detected: List[IssueItem] = Field(default_factory=list)
    quality_notes: QualityNotes = Field(default_factory=QualityNotes)


class PhoneItem(BaseModel):
    digits: str = Field(
        default="",
        description="RU contact phone digits only: 11 chars 7XXXXXXXXXX or 8XXXXXXXXXX, or 10 chars 9XXXXXXXXX",
    )
    speaker: int = Field(default=0, description="speaker id from transcript")
    evidence: str = Field(default="", description="exact short quote")


class CommitmentItem(BaseModel):
    who_spk: int = Field(default=0, description="speaker who promised")
    to_spk: int = Field(default=0, description="addressee speaker; 0 if unknown")
    promise: str = Field(default="", description="what was promised")
    when: str = Field(default="", description="deadline if said, else empty")
    evidence: str = Field(default="", description="exact short quote")


class AddressItem(BaseModel):
    text: str = Field(default="", description="address as said: city/street/house")
    speaker: int = Field(default=0)
    evidence: str = Field(default="", description="exact short quote")


class AmountItem(BaseModel):
    value: str = Field(default="", description="number as said, digits")
    currency: str = Field(default="", description="RUB/USD/EUR or empty")
    what: str = Field(default="", description="what the sum is for")
    speaker: int = Field(default=0)
    evidence: str = Field(default="", description="exact short quote")


class ExtractFactsResponse(BaseModel):
    call_id: str = Field(default="")
    phones: List[PhoneItem] = Field(max_length=4, description="contact phones only")
    addresses: List[AddressItem] = Field(max_length=4, description="visit/location addresses")
    amounts: List[AmountItem] = Field(max_length=6, description="money sums only")
    commitments: List[CommitmentItem] = Field(max_length=8, description="explicit promises")
    notes: str = Field(default="")


class SpeakerRoleItem(BaseModel):
    spk: int = Field(description="speaker id from transcript")
    role: str = Field(default="unknown", description="ivr, client, agent, or unknown")
    title: str = Field(default="", description="department or function if said")
    name: str = Field(default="", description="personal name if said")
    evidence: str = Field(default="", description="exact short quote")


class ExtractRolesResponse(BaseModel):
    call_id: str = Field(default="")
    speakers: List[SpeakerRoleItem] = Field(max_length=8, description="one row per speaker id")
    notes: str = Field(default="")


class BatchHighlightItem(BaseModel):
    text: str = Field(default="", description="grouped problem or positive moment")
    calls: List[str] = Field(default_factory=list, description="call_id list")
    count: int = Field(default=1, description="number of calls with this pattern")


class BatchRiskItem(BaseModel):
    risk: str = Field(default="")
    severity: str = Field(default="medium", description="low, medium, or high")
    evidence: str = Field(default="", description="short justification")
    calls: List[str] = Field(default_factory=list)


class BatchSummaryResponse(BaseModel):
    date: str = ""
    n_calls: int = 0
    executive_summary: str = Field(default="", description="3-5 sentence day overview in Russian")
    key_moments: List[str] = Field(default_factory=list, max_length=10)
    recurring_problems: List[BatchHighlightItem] = Field(default_factory=list, max_length=8)
    positive_moments: List[BatchHighlightItem] = Field(default_factory=list, max_length=6)
    potential_risks: List[BatchRiskItem] = Field(default_factory=list, max_length=8)
    top_topics: List[str] = Field(default_factory=list, max_length=8)
