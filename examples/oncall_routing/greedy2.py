# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional
from maro.simulator.scenarios.oncall_routing import Coordinate
from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing.common import Action, AllocateAction, OncallRoutingPayload, PostponeAction
from maro.utils import set_seeds

from examples.oncall_routing.utils import refresh_segment_index

set_seeds(512)


def _get_actions(running_env: Env, event: OncallRoutingPayload, oncall_order=None) -> List[Action]:
    tick = running_env.tick
    if oncall_order:
        oncall_orders = [oncall_order]
    else:
        oncall_orders = event.oncall_orders
    route_meta_info_dict = event.route_meta_info_dict
    route_plan_dict = event.route_plan_dict
    carriers_in_stop: List[bool] = (running_env.snapshot_list["carriers"][tick::"in_stop"] == 1).tolist()
    est_duration_predictor = event.estimated_duration_predictor

    actions = []
    for oncall_order in oncall_orders:
        min_duration = float("inf")
        chosen_route_name: Optional[str] = None
        insert_idx = -1

        for route_name, meta in route_meta_info_dict.items():
            carrier_idx = meta["carrier_idx"]
            planned_orders = route_plan_dict[route_name]

            for i, planned_order in enumerate(planned_orders):
                if i == 0 and not carriers_in_stop[carrier_idx]:
                    continue
                if i == 0 and carriers_in_stop[carrier_idx]:
                    current_staying_stop_coordinate: Coordinate = meta["current_staying_stop_coordinate"]
                    duration = est_duration_predictor.predict(tick, oncall_order.coord, planned_order.coord) \
                    + est_duration_predictor.predict(tick,oncall_order.coord, current_staying_stop_coordinate) \
                    - est_duration_predictor.predict(tick,planned_order.coord, current_staying_stop_coordinate)
                    if duration < min_duration:
                        is_time_valid = True
                        cur_tick = tick
                        estimated_next_departure_tick: int = meta["estimated_next_departure_tick"]
                        cur_tick = estimated_next_departure_tick
                        cur_tick += est_duration_predictor.predict(  # Current stop => oncall order
                                cur_tick, current_staying_stop_coordinate, oncall_order.coord)
                        #oncall_order.open_time-30 <=
                        if not (oncall_order.open_time <= cur_tick <= oncall_order.close_time):
                            is_time_valid = False
                        if is_time_valid:
                            for j in range(len(planned_orders)):
                                if j == 0:
                                    cur_tick += est_duration_predictor.predict(  # Current stop => oncall order
                                cur_tick, oncall_order.coord, planned_orders[j].coord)
                                else:
                                    cur_tick += est_duration_predictor.predict(  # Current stop => oncall order
                                cur_tick, planned_orders[j-1].coord, planned_orders[j].coord)
                                    #planned_orders[j].open_time-30 <=
                                if not (planned_orders[j].open_time <= cur_tick <= planned_orders[j].close_time):
                                    is_time_valid = False
                                    break
                        if is_time_valid:   
                            min_duration, chosen_route_name, insert_idx = duration, route_name, i                    
                if i < len(planned_orders)-1:
                    duration = est_duration_predictor.predict(tick, oncall_order.coord, planned_order.coord) \
                    + est_duration_predictor.predict(tick,oncall_order.coord, planned_orders[i+1].coord) \
                    - est_duration_predictor.predict(tick,planned_order.coord, planned_orders[i+1].coord)
                    if duration < min_duration:
                        is_time_valid = True
                        cur_tick = tick
                        estimated_next_departure_tick: int = meta["estimated_next_departure_tick"]
                        if carriers_in_stop[carrier_idx]:
                            cur_tick = estimated_next_departure_tick
                        cur_tick += meta["estimated_duration_to_the_next_stop"]
                        for j in range(1,i+1):
                            cur_tick += est_duration_predictor.predict(  # Current stop => oncall order
                                cur_tick, planned_orders[j-1].coord, planned_orders[j].coord)
                        cur_tick += est_duration_predictor.predict(  # Current stop => oncall order
                                cur_tick, planned_orders[i].coord, oncall_order.coord)
                        #oncall_order.open_time-30 <=
                        if not (oncall_order.open_time <= cur_tick <= oncall_order.close_time):
                            is_time_valid = False
                        if is_time_valid:
                            for j in range(i+1, len(planned_orders)):
                                if j == i+1:
                                    cur_tick += est_duration_predictor.predict(  # Current stop => oncall order
                                cur_tick, oncall_order.coord, planned_orders[j].coord)
                                else:
                                    cur_tick += est_duration_predictor.predict(  # Current stop => oncall order
                                cur_tick, planned_orders[j-1].coord, planned_orders[j].coord)
                                if not (planned_orders[j].open_time <= cur_tick <= planned_orders[j].close_time):
                                    is_time_valid = False
                                    break
                        if is_time_valid:   
                            min_duration, chosen_route_name, insert_idx = duration, route_name, i+1  

        if chosen_route_name is not None:
            actions.append(
                AllocateAction(order_id=oncall_order.id, route_name=chosen_route_name, insert_index=insert_idx)
            )
        else:
            for route_name, meta in route_meta_info_dict.items():
                carrier_idx = meta["carrier_idx"]
                planned_orders = route_plan_dict[route_name]
        
                for i, planned_order in enumerate(planned_orders):
                    if i == 0 and not carriers_in_stop[carrier_idx]:
                        continue
                    if i == 0 and carriers_in_stop[carrier_idx]:
                        current_staying_stop_coordinate: Coordinate = meta["current_staying_stop_coordinate"]
                        duration = est_duration_predictor.predict(tick, oncall_order.coord, planned_order.coord) \
                        + est_duration_predictor.predict(tick,oncall_order.coord, current_staying_stop_coordinate) \
                        - est_duration_predictor.predict(tick,planned_order.coord, current_staying_stop_coordinate)
                        if duration < min_duration:
                            min_duration, chosen_route_name, insert_idx = duration, route_name, i                    
                    if i < len(planned_orders)-1:
                        duration = est_duration_predictor.predict(tick, oncall_order.coord, planned_order.coord) \
                        + est_duration_predictor.predict(tick,oncall_order.coord, planned_orders[i+1].coord) \
                        - est_duration_predictor.predict(tick,planned_order.coord, planned_orders[i+1].coord)
                        if duration < min_duration:
                            min_duration, chosen_route_name, insert_idx = duration, route_name, i+1
            if chosen_route_name is not None:
                actions.append(
                    AllocateAction(order_id=oncall_order.id, route_name=chosen_route_name, insert_index=insert_idx)
                ) 
            else:
                actions.append(PostponeAction(order_id=oncall_order.id))

    actions = refresh_segment_index(actions)

    return actions


def get_greedy2_metrics(env):
    env.reset(keep_seed=True)
    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        assert isinstance(decision_event, OncallRoutingPayload)
        metrics, decision_event, is_done = env.step(_get_actions(env, decision_event))
    return metrics

# Greedy: assign each on-call order to the closest stop on existing route.
if __name__ == "__main__":
    env = Env(
        scenario="oncall_routing", topology="example", start_tick=480, durations=960,
    )
    #env._business_engine._random_seed = 1000
    env.reset(keep_seed=True)
    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        assert isinstance(decision_event, OncallRoutingPayload)
        print(f"Processing {len(decision_event.oncall_orders)} oncall orders at tick {env.tick}.")
        while decision_event and len(decision_event.oncall_orders) > 0:
            metrics, decision_event, is_done = env.step(_get_actions(env, decision_event, oncall_order=decision_event.oncall_orders[0]))

    print(metrics)
