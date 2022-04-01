# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from maro.simulator.scenarios.supply_chain.objects import SkuInfo
from maro.simulator.scenarios.supply_chain.units import DistributionUnit, ProductUnit, StorageUnit
from maro.simulator.scenarios.supply_chain.units.storage import StorageUnitInfo
from maro.simulator.scenarios.supply_chain.units.distribution import DistributionUnitInfo
from maro.simulator.scenarios.supply_chain.units.product import ProductUnitInfo

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain import UnitBase
    from maro.simulator.scenarios.supply_chain.datamodels.base import DataModelBase
    from maro.simulator.scenarios.supply_chain.world import World
    from maro.simulator.scenarios.supply_chain.units import UnitBase


@dataclass
class FacilityInfo:
    id: int
    name: str
    node_index: int
    class_name: type
    configs: dict
    skus: Dict[int, SkuInfo]
    upstreams: Dict[int, List[int]]
    downstreams: Dict[int, List[int]]  # Key: product_id; Value: facility id list
    storage_info: Optional[StorageUnitInfo]
    distribution_info: Optional[DistributionUnitInfo]
    products_info: Dict[int, ProductUnitInfo]  # Key: product_id


class FacilityBase(ABC):
    """Base of all facilities."""
    def __init__(
        self, id: int, name: str, data_model_name: str, data_model_index: int, world: World, config: dict
    ) -> None:
        # Id and name of this facility.
        self.id: int = id
        self.name: str = name

        # Name of data model, and the facility instance's corresponding node index in snapshot list.
        self.data_model_name: str = data_model_name
        self.data_model_index: int = data_model_index

        # World of this facility belongs to.
        self.world: World = world

        # Configuration of this facility.
        self.configs: dict = config

        # SKUs in this facility.
        self.skus: Dict[int, SkuInfo] = {}

        # Product units for each sku in this facility.
        # Key is sku(product) id, value is the instance of product unit.
        self.products: Optional[Dict[int, ProductUnit]] = None

        # Storage unit in this facility.
        self.storage: Optional[StorageUnit] = None

        # Distribution unit in this facility.
        self.distribution: Optional[DistributionUnit] = None

        # Upstream facilities.
        # Key is sku id, value is the list of facilities from upstream.
        self.upstreams: Dict[int, List[FacilityBase]] = defaultdict(list)

        # Down stream facilities, value same as upstreams.
        self.downstreams: Dict[int, List[FacilityBase]] = defaultdict(list)

        self.data_model: Optional[DataModelBase] = None

        # Children of this facility (valid units).
        self.children: List[UnitBase] = []

    def parse_skus(self, configs: dict) -> None:
        """Parse sku information from config.

        Args:
            configs (dict): Configuration of skus belongs to this facility.
        """
        for sku_id_or_name, sku_config in configs.items():
            sku_id, sku_name = self.world.get_sku_id_and_name(sku_id_or_name)
            sku_config['id'] = sku_id
            sku_config['name'] = sku_name
            facility_sku = SkuInfo(**sku_config)
            self.skus[facility_sku.id] = facility_sku

    def get_config(self, key: str, default: object = None) -> object:
        """Get specified configuration of facility.

        Args:
            key (str): Key of the configuration.
            default (object): Default value if key not exist, default is None.

        Returns:
            object: value in configuration.
        """
        return default if self.configs is None else self.configs.get(key, default)

    def initialize(self) -> None:
        """Initialize this facility after frame is ready."""
        self.data_model.initialize()

        # Put valid units into the children, used to simplify following usage.
        if self.storage is not None:
            self.children.append(self.storage)

        if self.distribution is not None:
            self.children.append(self.distribution)

        if self.products is not None:
            for product in self.products.values():
                self.children.append(product)

    def step(self, tick: int) -> None:
        """Push facility to next step.

        Args:
            tick (int): Current simulator tick.
        """
        for unit in self.children:
            unit.step(tick)

    # TODO: confirm Why not call flush_states() immediately after each unit.step()?
    def flush_states(self) -> None:
        """Flush states into frame."""
        for unit in self.children:
            unit.flush_states()

    def post_step(self, tick: int) -> None:
        """Post processing at the end of step."""
        for unit in self.children:
            unit.post_step(tick)

    def reset(self) -> None:
        """Reset facility for new episode."""
        for unit in self.children:
            unit.reset()

        if self.data_model is not None:
            self.data_model.reset()

    def get_in_transit_orders(self) -> dict:
        in_transit_orders = defaultdict(int)

        for product_id, product in self.products.items():
            if product.consumer is not None:
                in_transit_orders[product_id] = product.consumer.get_in_transit_quantity()

        return in_transit_orders

    def set_action(self, action: object) -> None:
        pass

    def get_node_info(self) -> FacilityInfo:
        return FacilityInfo(
            id=self.id,
            name=self.name,
            node_index=self.data_model_index,
            class_name=type(self),
            configs=self.configs,
            skus=self.skus,
            upstreams={product_id: [f.id for f in f_list] for product_id, f_list in self.upstreams.items()},
            downstreams={product_id: [f.id for f in f_list] for product_id, f_list in self.downstreams.items()},
            storage_info=self.storage.get_unit_info() if self.storage else None,
            distribution_info=self.distribution.get_unit_info() if self.distribution else None,
            products_info={product_id: product.get_unit_info() for product_id, product in self.products.items()},
        )
