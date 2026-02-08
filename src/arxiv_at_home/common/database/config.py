from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    connection_url: str
