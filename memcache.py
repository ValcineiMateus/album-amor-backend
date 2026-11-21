"""Cache de um valor com validade e "voo único" (single-flight).

Usado para a lista de memórias do álbum: antes, cada /photos e cada mensagem da IA liam a coleção
inteira do Firestore. Agora o resultado vale por `ttl` segundos, e se várias requisições chegam ao
mesmo tempo só UMA vai ao Firestore (as outras esperam o mesmo resultado).
"""
import asyncio
import time
from typing import Awaitable, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class SingleFlightCache(Generic[T]):
    def __init__(self, ttl: float, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self._clock = clock
        self._lock = asyncio.Lock()
        self._value: Optional[T] = None
        self._has_value = False
        self._expires = 0.0
        self._gen = 0  # aumenta a cada invalidação, para não gravar uma leitura que ficou velha no meio do caminho
        self.loads = 0  # quantas vezes realmente foi à fonte (útil em testes/diagnóstico)

    def _fresh(self) -> bool:
        return self._has_value and self._clock() < self._expires

    async def get(self, loader: Callable[[], Awaitable[T]]) -> T:
        if self._fresh():
            return self._value  # type: ignore[return-value]
        async with self._lock:
            if self._fresh():  # outra requisição acabou de carregar enquanto esta esperava
                return self._value  # type: ignore[return-value]
            gen = self._gen
            value = await loader()
            self.loads += 1
            if gen == self._gen:
                self._value, self._has_value, self._expires = value, True, self._clock() + self.ttl
            return value

    def invalidate(self) -> None:
        self._gen += 1
        self._has_value = False
        self._expires = 0.0
