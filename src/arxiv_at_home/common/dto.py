import datetime as dt

from pydantic import BaseModel


class PaperMetadataVersion(BaseModel):
    version: str
    created: dt.datetime


class PaperMetadata(BaseModel):
    source: str

    id: str
    authors: str
    title: str
    doi: str | None
    license: str | None
    abstract: str
    categories: set[str]
    journal_ref: str | None
    updated_at: dt.datetime
    versions: list[PaperMetadataVersion]

    @property
    def fully_qualified_name(self) -> str:
        return f"{self.source}/{self.id}"
