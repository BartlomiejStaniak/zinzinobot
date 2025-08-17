import json
from datetime import datetime, timedelta
from typing import Dict, Any, List


class MemoryManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_store = {}

    async def store_data(self, category: str, data: Dict[str, Any]):
        if category not in self.memory_store:
            self.memory_store[category] = []
        self.memory_store[category].append(data)

    async def get_data(self, category: str, filters: Dict[str, Any] = None) -> List[Dict]:
        if category not in self.memory_store:
            return []

        data = self.memory_store[category]
        if filters:
            # Prosta filtracja
            filtered = []
            for item in data:
                match = all(item.get(k) == v for k, v in filters.items())
                if match:
                    filtered.append(item)
            return filtered
        return data

    async def get_context(self, context_type: str) -> Dict[str, Any]:
        return self.memory_store.get(context_type, {})

    async def get_recent_data(self, category: str, days: int = 7) -> List[Dict]:
        data = self.memory_store.get(category, [])
        cutoff = datetime.now() - timedelta(days=days)

        recent = []
        for item in data:
            if 'created_at' in item:
                created = datetime.fromisoformat(item['created_at'])
                if created > cutoff:
                    recent.append(item)
        return recent

    async def get_usage_stats(self) -> float:
        # Symulacja użycia pamięci
        return 0.45  # 45%

    async def cleanup(self):
        # Czyszczenie starych danych
        pass

    async def optimize(self):
        # Optymalizacja pamięci
        pass