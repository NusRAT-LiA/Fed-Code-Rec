
"""Data repository layer"""
from typing import List, Optional, TypeVar, Generic
from abc import ABC, abstractmethod

T = TypeVar('T')

class Repository(ABC, Generic[T]):
    """Generic repository interface"""
    
    @abstractmethod
    def save(self, entity: T) -> T:
        """Save entity"""
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id: int) -> Optional[T]:
        """Find entity by ID"""
        pass
    
    @abstractmethod
    def find_all(self) -> List[T]:
        """Find all entities"""
        pass
    
    @abstractmethod
    def update(self, entity: T) -> bool:
        """Update entity"""
        pass
    
    @abstractmethod
    def delete(self, entity_id: int) -> bool:
        """Delete entity"""
        pass

class InMemoryRepository(Repository[T]):
    """In-memory repository implementation"""
    
    def __init__(self):
        self.storage = {}
        self.next_id = 1
    
    def save(self, entity: T) -> T:
        """Save entity to memory"""
        entity_id = getattr(entity, 'id', self.next_id)
        self.storage[entity_id] = entity
        self.next_id += 1
        return entity
    
    def find_by_id(self, entity_id: int) -> Optional[T]:
        """Find entity by ID"""
        return self.storage.get(entity_id)
    
    def find_all(self) -> List[T]:
        """Find all entities"""
        return list(self.storage.values())
    
    def update(self, entity: T) -> bool:
        """Update entity"""
        entity_id = getattr(entity, 'id', None)
        if entity_id and entity_id in self.storage:
            self.storage[entity_id] = entity
            return True
        return False
    
    def delete(self, entity_id: int) -> bool:
        """Delete entity"""
        if entity_id in self.storage:
            del self.storage[entity_id]
            return True
        return False

class TransactionManager:
    """Manage database transactions"""
    
    def __init__(self):
        self.active = False
    
    def begin_transaction(self):
        """Begin new transaction"""
        self.active = True
    
    def commit(self):
        """Commit transaction"""
        self.active = False
    
    def rollback(self):
        """Rollback transaction"""
        self.active = False
