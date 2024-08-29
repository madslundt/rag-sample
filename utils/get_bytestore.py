from langchain_core.stores import InMemoryByteStore


def get_bytestore() -> InMemoryByteStore:
    return InMemoryByteStore()
