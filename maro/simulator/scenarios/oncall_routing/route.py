# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

from maro.backends.frame import NodeBase, node

from . import Order
from .plan_element import PlanElement


@node("routes")
class Route(NodeBase):

    def __init__(self) -> None:
        self._name = None
        self._carrier_idx = None
        self._carrier_name = None

        self.finished_orders: List[Tuple[Order, int]] = []
        self.remaining_plan: List[PlanElement] = []

    @property
    def idx(self) -> int:
        return self.index

    @property
    def name(self) -> str:
        return self._name

    @property
    def carrier_idx(self) -> int:
        return self._carrier_idx

    @property
    def carrier_name(self) -> str:
        return self._carrier_name

    def set_init_state(self, name: str, carrier_idx: int, carrier_name: str) -> None:
        self._name = name
        self._carrier_idx = carrier_idx
        self._carrier_name = carrier_name

        self.reset()

    def reset(self) -> None:
        pass
