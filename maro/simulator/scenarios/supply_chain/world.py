# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
import itertools
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from maro.backends.frame import FrameBase

from .facilities import FacilityBase
from .frame_builder import build_frame
from .objects import SkuInfo, SkuMeta
from .parser import DataModelDef, EntityDef, SupplyChainConfiguration
from .units import ExtendUnitBase, ProductUnit, UnitBase


@dataclass
class SupplyChainEntity:
    id: int
    class_type: type
    skus: Optional[SkuInfo]
    facility_id: int
    parent_id: Optional[int]

    @property
    def is_facility(self) -> bool:
        return issubclass(self.class_type, FacilityBase)


class World:
    """Supply chain world contains facilities."""

    def __init__(self) -> None:
        # Frame for current world configuration.
        self.frame: Optional[FrameBase] = None

        # Current configuration.
        self.configs: Optional[SupplyChainConfiguration] = None

        # Durations of current simulation.
        self.durations = 0

        # All the entities in the world.
        self.units: Dict[int, UnitBase] = {}

        # All the facilities in this world.
        self.facilities: Dict[int, FacilityBase] = {}

        # Entity id counter, every unit and facility have unique id.
        self._id_counter = itertools.count(1)

        # Sku name to id mapping, used for querying.
        self._sku_name2id_mapping = {}

        # All the sku in this world.
        self._sku_collection: Dict[int, SkuMeta] = {}

        # Facility name to id mapping, used for querying.
        self._facility_name2id_mapping = {}

        # Data model class collection, used to collection data model class and their number in frame.
        self._data_class_collection = collections.defaultdict(lambda: 0)

        self.entity_list = []

        self.max_sources_per_facility = 0
        self.max_price = 0

    def get_sku_by_name(self, name: str) -> SkuMeta:
        """Get sku information by name.

        Args:
            name (str): Sku name to query.

        Returns:
            SkuMeta: Meta information for sku.
        """
        return self._sku_collection[self._sku_name2id_mapping[name]]

    def get_sku_by_id(self, sku_id: int) -> SkuMeta:
        """Get sku information by sku id.

        Args:
            sku_id (int): Id of sku to query.

        Returns:
            SkuMeta: Meta information for sku.
        """
        return self._sku_collection[sku_id]

    def get_sku_id_by_name(self, name: str) -> int:
        return self._sku_name2id_mapping[name]

    def get_facility_by_id(self, facility_id: int) -> FacilityBase:
        """Get facility by id.

        Args:
            facility_id (int): Facility id to query.

        Returns:
            FacilityBase: Facility instance.
        """
        return self.facilities[facility_id]

    def get_facility_by_name(self, name: str) -> FacilityBase:
        """Get facility by name.

        Args:
            name (str): Facility name to query.

        Returns:
            FacilityBase: Facility instance.
        """
        return self.facilities[self._facility_name2id_mapping[name]]

    def get_entity_by_id(self, entity_id: int) -> Union[FacilityBase, UnitBase]:
        """Get an entity (Unit or Facility) by id.

        Args:
            entity_id (int): Id to query.

        Returns:
            Union[FacilityBase, UnitBase]: Unit or facility instance.
        """
        return self.units[entity_id] if entity_id in self.units else self.facilities[entity_id]

    def get_sku_id_and_name(self, id_or_name: Union[int, str]) -> Tuple[int, str]:
        if isinstance(id_or_name, int):
            assert id_or_name in self._sku_collection.keys()
            return id_or_name, self._sku_collection[id_or_name].name
        else:
            assert id_or_name in self._sku_name2id_mapping.keys()
            return self._sku_name2id_mapping[id_or_name], id_or_name

    def _init_sku_collection(self) -> None:
        # Grab sku information for this world.
        for sku_conf in self.configs.world["skus"]:
            sku = SkuMeta(**sku_conf)

            self._sku_name2id_mapping[sku.name] = sku.id
            self._sku_collection[sku.id] = sku

        # Format bom info to use sku id as key.
        for sku_conf in self.configs.world["skus"]:
            sku = self._sku_collection[sku_conf["id"]]
            sku.bom = {}

            bom = sku_conf.get("bom", {})
            for src_id_or_name, units_per_lot in bom.items():
                src_id, _ = self.get_sku_id_and_name(src_id_or_name)
                sku.bom[src_id] = units_per_lot

    def _create_entities(self) -> None:
        for facility_conf in self.configs.world["facilities"]:
            facility_def: EntityDef = self.configs.entity_defs[facility_conf["class"]]
            assert issubclass(facility_def.class_type, FacilityBase)

            # Due with data model.
            data_model_def: DataModelDef = self.configs.data_model_defs[facility_def.data_model_alias]
            # Register the data model, so that it will help to generate related instance index.
            data_model_index = self._register_data_model(data_model_def.alias)

            # Instance of facility.
            facility: FacilityBase = facility_def.class_type(
                id=self._gen_id(),
                name=facility_conf["name"],
                data_model_name=data_model_def.name_in_frame,
                data_model_index=data_model_index,
                world=self,
                config=facility_conf.get("config", {})
            )

            # Parse sku info.
            facility.parse_skus(facility_conf["skus"])

            # Build children Units.
            for child_name, child_conf in facility_conf["children"].items():
                child = self._build_unit(facility=facility, parent=facility, config=child_conf)
                setattr(facility, child_name, child)

            # Build ProductUnits.
            if "products" in facility_conf:
                products_dict = self._build_product_units(facility=facility, config=facility_conf["products"])
                setattr(facility, "products", products_dict)

            self.facilities[facility.id] = facility
            self._facility_name2id_mapping[facility.name] = facility.id

    def _build_frame_and_assgin_to_entities(self, snapshot_number: int) -> None:
        self.frame = self._build_frame(snapshot_number)

        for unit in self.units.values():
            if unit.data_model_name is not None:
                unit.data_model = getattr(self.frame, unit.data_model_name)[unit.data_model_index]

        for facility in self.facilities.values():
            if facility.data_model_name is not None:
                facility.data_model = getattr(self.frame, facility.data_model_name)[facility.data_model_index]

    # TODO: build from yaml + build from input files?
    def build(self, configs: SupplyChainConfiguration, snapshot_number: int, durations: int) -> None:
        """Build world with configurations.

        Args:
            configs (SupplyChainConfiguration): Configuration of current world.
            snapshot_number (int): Number of snapshots to keep in memory.
            durations (int): Durations of current simulation.
        """
        self.durations = durations
        self.configs = configs

        # Step 1: Initialize sku collection based on the world.skus in config.
        self._init_sku_collection()

        # Step 2: Create FacilityBase instances & its child UnitBases instances based on the world.facilities in config.
        self._create_entities()

        # Step 3: Build frame and assign data model instances to entities.
        self._build_frame_and_assgin_to_entities(snapshot_number)

        # Construct the upstream topology.
        topology = self.configs.world["topology"]

        for cur_facility_name, topology_conf in topology.items():
            facility = self.get_facility_by_name(cur_facility_name)

            for sku_name, source_facilities in topology_conf.items():
                sku = self.get_sku_by_name(sku_name)
                facility.upstreams[sku.id] = []

                self.max_sources_per_facility = max(self.max_sources_per_facility, len(source_facilities))

                for source_name in source_facilities:
                    source_facility = self.get_facility_by_name(source_name)
                    facility.upstreams[sku.id].append(source_facility)
                    source_facility.downstreams[sku.id].append(facility)

        # Call initialize method for facilities.
        for facility in self.facilities.values():
            facility.initialize()

        # Call initialize method for units.
        for unit in self.units.values():
            unit.initialize()

        # Collection entity list
        for facility in self.facilities.values():
            entity = SupplyChainEntity(
                id=facility.id, class_type=facility.__class__, skus=None, facility_id=facility.id, parent_id=None,
            )
            self.entity_list.append(entity)

        for unit in self.units.values():
            entity = SupplyChainEntity(
                id=unit.id, class_type=unit.__class__,
                skus=unit.facility.skus[unit.product_id] if isinstance(unit, ExtendUnitBase) else None,
                facility_id=unit.facility.id, parent_id=unit.parent.id,
            )
            self.entity_list.append(entity)

    def _build_product_unit(
        self, unit_def: EntityDef, parent: Union[FacilityBase, UnitBase], facility: FacilityBase, config: dict,
    ) -> None:
        """Build an unit by its type.

        Args:
            unit_def (EntityDef): Definition of this unit.
            parent (Union[FacilityBase, UnitBase]): Parent of this unit.
            facility (FacilityBase): Facility this unit belongs to.

        Returns:
            UnitBase: Unit instance.
        """
        assert issubclass(unit_def.class_type, UnitBase)

        data_model_name, data_model_index = None, None
        if unit_def.data_model_alias is not None:
            # Due with data model.
            data_model_def: DataModelDef = self.configs.data_model_defs[unit_def.data_model_alias]

            data_model_name = data_model_def.name_in_frame

            # Register the data model, so that it will help to generate related instance index.
            data_model_index = self._register_data_model(data_model_def.alias)

        unit = unit_def.class_type(
            id=self._gen_id(),
            data_model_name=data_model_name,
            data_model_index=data_model_index,
            facility=facility,
            parent=parent,
            world=self,
            config=config,
        )

        self.units[unit.id] = unit

        return unit

    def _build_unit(
        self, facility: FacilityBase, parent: Union[FacilityBase, UnitBase], config: dict,
    ) -> UnitBase:
        """Build an unit by its configuration.

        Args:
            facility (FacilityBase): Facility of this unit belongs to.
            parent (Union[FacilityBase, UnitBase]): Parent of this unit belongs to, could be a facility or a unit.
            config (dict): Configuration of this unit.

        Returns:
            Optional[UnitBase, Dict[int, ProductUnit]]: An UnitBase instance or a dict of ProductUnit.
        """
        unit_def: EntityDef = self.configs.entity_defs[config["class"]]
        assert issubclass(unit_def.class_type, UnitBase)

        # Due with data model.
        data_model_def: DataModelDef = self.configs.data_model_defs[unit_def.data_model_alias]
        # Register the data model, so that it will help to generate related instance index.
        data_model_index = self._register_data_model(data_model_def.alias)

        unit_instance = unit_def.class_type(
            id=self._gen_id(),
            data_model_name=data_model_def.name_in_frame,
            data_model_index=data_model_index,
            facility=facility,
            parent=parent,
            world=self,
            config=config.get("config", {})
        )

        # Prepare children.
        children_conf = config.get("children", None)

        if children_conf:
            unit_instance.children = []

            for child_name, child_conf in children_conf.items():
                # If child configuration is a dict, then we add it as a property by name (key).
                if type(child_conf) == dict:
                    child_instance = self._build_unit(facility, unit_instance, child_conf)

                    setattr(unit_instance, child_name, child_instance)
                    unit_instance.children.append(child_instance)

                elif type(child_conf) == list:
                    # If child configuration is a list, then will treat it as list property, named same as key.
                    child_list = []
                    for conf in child_conf:
                        child_list.append(self._build_unit(facility, unit_instance, conf))

                    setattr(unit_instance, child_name, child_list)
                    unit_instance.children.extend(child_list)

        # Record the id.
        self.units[unit_instance.id] = unit_instance

        return unit_instance

    def _build_product_units(self, facility: FacilityBase, config: dict) -> Dict[int, ProductUnit]:
        """Generate product unit by sku information.

        Args:
            facility (FacilityBase): Facility this product belongs to.
            config (dict): Config of children unit.
            unit_def (object): Definition of the unit (from config).

        Returns:
            dict: Dictionary of product unit, key is the product id, value is ProductUnit.
        """
        unit_def: EntityDef = self.configs.entity_defs[config["class"]]
        assert issubclass(unit_def.class_type, ProductUnit)

        config = config.get("config")

        products_dict: Dict[int, ProductUnit] = {}

        if facility.skus is not None and len(facility.skus) > 0:
            for sku_id, sku in facility.skus.items():
                sku_type = sku.type

                product_unit: ProductUnit = self._build_product_unit(unit_def, facility, facility, config)
                product_unit.product_id = sku_id
                product_unit.children = []
                product_unit.storage = product_unit.facility.storage
                product_unit.distribution = product_unit.facility.distribution

                # NOTE: BE CAREFUL about the order, product unit will use this order update children,
                # the order may affect the states.
                # Here we make sure consumer is the first one, so it can place order first.
                for child_name in ("consumer", "seller", "manufacture"):
                    conf = config.get(child_name, None)

                    if conf is not None:
                        # Ignore manufacture unit if it is not for a production, even it is configured in config.
                        if sku_type != "production" and child_name == "manufacture":
                            continue

                        # We produce the product, so we do not need to purchase it.
                        if sku_type == "production" and child_name == "consumer":
                            continue

                        child_unit = self._build_unit(facility, product_unit, conf)
                        child_unit.product_id = sku_id

                        setattr(product_unit, child_name, child_unit)

                        product_unit.children.append(child_unit)

                products_dict[sku_id] = product_unit

        return products_dict

    def get_node_mapping(self) -> dict:
        """Collect all the entities' information.

        Returns:
            dict: A dictionary contains 'mapping' for id to data model index mapping,
                'detail' for detail of units and facilities.
        """
        facility_info_dict = {
            facility_id: facility.get_node_info() for facility_id, facility in self.facilities.items()
        }

        id2index_mapping = {}

        for unit_id, unit in self.units.items():
            sku = None

            if isinstance(unit, ExtendUnitBase):
                sku = unit.facility.skus[unit.product_id]

            if unit.data_model is not None:
                # TODO: replace with data class or named tuple
                id2index_mapping[unit_id] = (unit.data_model_name, unit.data_model_index, unit.facility.id, sku)
            else:
                id2index_mapping[unit_id] = (None, None, unit.facility.id, sku)

        return {
            "unit_mapping": id2index_mapping,
            "skus": {id_: sku for id_, sku in self._sku_collection.items()},
            "facilities": facility_info_dict,
            "max_price": self.max_price,
            "max_sources_per_facility": self.max_sources_per_facility,
        }

    def _register_data_model(self, alias: str) -> int:
        """Register a data model alias, used to collect data model used in frame.

        Args:
            alias (str): Class alias defined in core.yml.

        Returns:
            int: Specified data model instance index after frame is built.
        """
        self._data_class_collection[alias] += 1
        return self._data_class_collection[alias] - 1

    def _build_frame(self, snapshot_number: int) -> FrameBase:
        """Build frame by current world definitions.

        Args:
            snapshot_number (int): Number of snapshots to keep in memory.

        Returns:
            FrameBase: The frame instance with data model in current configuration.
        """
        data_class_in_frame = []

        for alias, number in self._data_class_collection.items():
            data_model_def: DataModelDef = self.configs.data_model_defs[alias]
            data_class_in_frame.append((
                data_model_def.class_type,
                data_model_def.name_in_frame,
                number,
            ))

        frame = build_frame(True, snapshot_number, data_class_in_frame)

        return frame

    def _gen_id(self) -> int:
        """Generate id for entities."""
        return next(self._id_counter)
