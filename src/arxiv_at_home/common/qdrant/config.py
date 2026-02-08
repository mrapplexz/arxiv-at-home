from pydantic import BaseModel


class QdrantConfig(BaseModel):
    host: str
    grpc_port: int
