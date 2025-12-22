
"""Core business logic layer"""
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Order:
    """Order entity"""
    order_id: int
    customer_id: int
    items: List[dict]
    total_amount: float
    status: str
    created_at: datetime

class OrderService(ABC):
    """Abstract order service interface"""
    
    @abstractmethod
    def create_order(self, customer_id: int, items: List[dict]) -> Order:
        """Create new order"""
        pass
    
    @abstractmethod
    def get_order(self, order_id: int) -> Optional[Order]:
        """Retrieve order by ID"""
        pass
    
    @abstractmethod
    def update_order_status(self, order_id: int, status: str) -> bool:
        """Update order status"""
        pass

class OrderServiceImpl(OrderService):
    """Order service implementation"""
    
    def __init__(self, repository):
        self.repository = repository
    
    def create_order(self, customer_id: int, items: List[dict]) -> Order:
        """Create new order"""
        total = sum(item['price'] * item['quantity'] for item in items)
        
        order = Order(
            order_id=self._generate_id(),
            customer_id=customer_id,
            items=items,
            total_amount=total,
            status='pending',
            created_at=datetime.now()
        )
        
        self.repository.save(order)
        return order
    
    def get_order(self, order_id: int) -> Optional[Order]:
        """Retrieve order by ID"""
        return self.repository.find_by_id(order_id)
    
    def update_order_status(self, order_id: int, status: str) -> bool:
        """Update order status"""
        order = self.get_order(order_id)
        if order:
            order.status = status
            self.repository.update(order)
            return True
        return False
    
    def _generate_id(self) -> int:
        """Generate unique order ID"""
        return hash(datetime.now()) % 1000000

def calculate_discount(order: Order, discount_rate: float) -> float:
    """Calculate discount for order"""
    return order.total_amount * discount_rate
