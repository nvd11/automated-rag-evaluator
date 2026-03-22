import pytest
from src.dao.pgvector_retriever_dao import PgVectorRetrieverDAO
from src.domain.models import SearchQuery, RetrievedContext
import asyncio

def test_retriever_dao_initialization():
    dao = PgVectorRetrieverDAO()
    assert isinstance(dao, PgVectorRetrieverDAO)
