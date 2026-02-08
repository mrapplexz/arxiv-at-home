from pydantic import BaseModel


class RerankerConfig(BaseModel):
    device: str
    model: str
    template: str

    token_true: str
    token_false: str
