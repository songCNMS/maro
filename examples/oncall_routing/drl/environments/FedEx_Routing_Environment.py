# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
sys.path.append('/data/songlei/maro/')
import random
from collections import namedtuple
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from typing import List, Optional

from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing import Coordinate
from maro.simulator.scenarios.oncall_routing.common import AllocateAction, PostponeAction, OncallRoutingPayload, Order
from maro.utils import set_seeds
from examples.oncall_routing.utils import refresh_segment_index

from examples.oncall_routing.greedy import get_greedy_metrics
from examples.oncall_routing.greedy_with_time_window import get_greedy_with_time_window_metrics
from examples.oncall_routing.greedy2 import get_greedy2_metrics


NUM_ROUTE = 4
NUM_ONCALL_ORDERS = 6
NO_VALID_PENALTY_COEF = 50.0

REWARD_NORM = 1000.0
LAT_NORM = 50
LONG_NORM = 150
DURATION_NORM = 100.0

MAX_NUM_ROUTE = 130
MAX_NUM_STOPS_PER_ROUTE = 80


class Routing_Environment(gym.Env):
    environment_name = "FedEx Routing Environment"

    def __init__(self):
        self.action_space = spaces.Discrete(NUM_ROUTE+1)
        self.reward_threshold = float("-inf")
        self.trials = 100
        self._max_episode_steps = 960
        self.id = "FedEx Routing"
        self.running_env = Env(scenario="oncall_routing", topology="example_clean", start_tick=480, durations=self._max_episode_steps)
        self.before_buffer = self.running_env.configs['order_transition']['buffer_before_open_time']
        self.after_buffer = self.running_env.configs['order_transition']['buffer_after_open_time']
        
        self.route_len_mean = {}
        self.route_cnt = {}
        self.reset()
        self.num_states = len(self.state) - len(self.action_mask)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_states,), dtype=np.float32)


    def seed(self, s):
        set_seeds(s)

    def print_eval_result(self, metrics):
        finish_ticks = list(metrics['route_finish_tick'].values())
        max_tick, mean_tick, min_tick, sum_tick, std_tick = np.max(finish_ticks), np.mean(finish_ticks), np.min(finish_ticks), np.sum(finish_ticks), np.std(finish_ticks)
        print(metrics)
        print(f"\r total oncall: {metrics['total_oncall_num']}, \
                total order: {metrics['total_order_num']}, \
                    total delayed: {metrics['total_order_delayed']}, \
                        total delayed time: {self.total_delayed_tick},\r \
                            total terminated: {metrics['total_order_terminated']}, \
                                total terminated time: {self.total_terminated_tick},\
                                    tick: {min_tick}, {mean_tick}, {max_tick} \r")
        sys.stdout.flush()
        res = {"total_tick": sum_tick,
               "max_tick": max_tick,
               "std_tick": std_tick, 
               "total_order": metrics['total_order_num'],
               "total_oncall": metrics['total_oncall_num'],
               "total_in_advanced": metrics['total_order_in_advance'],
               "total_delayed": metrics["total_order_delayed"],
               "total_terminated": metrics["total_order_terminated"]}
        return res

    def get_greedy_metrics_env(self):
        metrics = get_greedy_metrics(self.running_env)
        return self.print_eval_result(metrics)

    def get_greedy_with_time_window_metrics_env(self):
        metrics = get_greedy_with_time_window_metrics(self.running_env)
        return self.print_eval_result(metrics)
    
    def get_greedy2_metrics_env(self):
        metrics = get_greedy2_metrics(self.running_env)
        return self.print_eval_result(metrics)
    
    def reset(self, keep_seed=False):
        self.episode_steps = 0
        self.running_env.reset(keep_seed=keep_seed)
        self.metrics, self.event, self.done = self.running_env.step(None)
        oncall_orders = self.event.oncall_orders
        # self.state = np.zeros(self.num_states)
        self.cur_oncall_order = None
        self.action_mask = np.zeros(NUM_ROUTE+1)
        # if len(oncall_orders) > 0:
        self.cur_oncall_order = oncall_orders[0]
        self.state = self.get_state()
        self.total_delayed_tick = 0
        self.total_terminated_tick = 0
        self.total_delayed_num = 0
        self.total_terminated_num = 0
        return self.state

    def get_actions_on_route(self, route_name):
        tick = self.running_env.tick
        oncall_order = self.cur_oncall_order
        route_meta_info_dict = self.event.route_meta_info_dict
        meta = route_meta_info_dict[route_name]
        route_plan_dict = self.event.route_plan_dict
        carriers_in_stop: List[bool] = (self.running_env.snapshot_list["carriers"][tick::"in_stop"] == 1).tolist()
        est_duration_predictor = self.event.estimated_duration_predictor

        num_order = len(route_plan_dict[route_name])
        route_original_indexes = list(range(num_order))

        # Best result with violating time windows
        min_duration_violate = float("inf")
        chosen_route_name_violate: Optional[str] = None
        insert_idx_violate = -1

        # Best result without violating any time windows
        min_duration_no_violate = float("inf")
        chosen_route_name_no_violate: Optional[str] = None
        insert_idx_no_violate = -1

        carrier_idx = meta["carrier_idx"]
        estimated_next_departure_tick: int = meta["estimated_next_departure_tick"]
        planned_orders = route_plan_dict[route_name]
        
        previous_coord = None
        next_coord = None
        duration_incr = 0

        for i, planned_order in enumerate(planned_orders):  # To traverse the insert index
            if i == 0 and not carriers_in_stop[carrier_idx]:
                continue
            duration = None
            # Check if it will violate any time window
            is_time_valid = True
            cur_tick = tick
            for j in range(len(planned_orders)):  # Simulate all orders
                if j == i:  # If we need to insert the oncall order before the j'th planned order
                    if j == 0:  # Carrier in stop. Insert before the first stop.
                        current_staying_stop_coordinate: Coordinate = meta["current_staying_stop_coordinate"]
                        cur_tick += estimated_next_departure_tick
                        duration = est_duration_predictor.predict(  # Current stop => oncall order
                            cur_tick, current_staying_stop_coordinate, oncall_order.coord)
                        oncall_order_closed_tick = cur_tick + duration
                        duration -= est_duration_predictor.predict(cur_tick, current_staying_stop_coordinate, planned_orders[j].coord)
                        cur_tick = oncall_order_closed_tick
                    else:
                        duration = est_duration_predictor.predict(  # Last planned order => oncall order
                            cur_tick, planned_orders[j - 1].coord, oncall_order.coord
                        )
                        oncall_order_closed_tick = cur_tick + duration
                        duration -= est_duration_predictor.predict(cur_tick, planned_orders[j-1].coord, planned_orders[j].coord)
                        cur_tick = oncall_order_closed_tick
                    cur_duration = est_duration_predictor.predict(cur_tick, oncall_order.coord, planned_order.coord)
                    duration += cur_duration
                    cur_tick += cur_duration  # Oncall order => current planned order
                    # Check if violate the oncall order time window or not
                    # if not oncall_order.open_time-self.before_buffer*1.5 <= cur_tick <= oncall_order.close_time+self.after_buffer*1.5:
                    if (oncall_order_closed_tick > oncall_order.close_time 
                        or oncall_order_closed_tick < oncall_order.open_time
                        or cur_tick < planned_order.open_time - self.before_buffer
                        or cur_tick > planned_order.close_time+duration+self.after_buffer):
                        is_time_valid = False
                        break
                else:
                    if j == 0:
                        # Current position (on the way or in a stop) => first planned order
                        if carriers_in_stop[carrier_idx]:
                            cur_tick = estimated_next_departure_tick
                        cur_tick += meta["estimated_duration_to_the_next_stop"]
                    else:
                        # Last planned order => current planned order
                        cur_tick += est_duration_predictor.predict(
                            cur_tick, planned_orders[j - 1].coord, planned_orders[j].coord
                        )

                    # Violate current planned order time window
                    # if not planned_orders[j].open_time - self.before_buffer*1.5 <= cur_tick <= planned_orders[j].close_time+self.after_buffer*1.5:
                    if j > i and not (planned_orders[j].open_time - self.before_buffer <= cur_tick <= planned_orders[j].close_time+duration+self.after_buffer):
                        is_time_valid = False
                        break
            if not duration:
                continue

            if is_time_valid:
                if duration < min_duration_no_violate:
                    min_duration_no_violate = duration
                    insert_idx_no_violate = i
                    chosen_route_name_no_violate = route_name
            if duration < min_duration_violate:
                min_duration_violate = duration
                insert_idx_violate = i
                chosen_route_name_violate = route_name

        is_valid = -1
        insert_idx = 0
        if chosen_route_name_no_violate is not None:
            if insert_idx_no_violate == 0:
                previous_coord = oncall_order.coord
            else:
                previous_coord = planned_orders[insert_idx_no_violate-1].coord
            next_coord = planned_orders[insert_idx_no_violate].coord
            duration_incr = min_duration_no_violate
            is_valid = 1
            insert_idx = insert_idx_no_violate
        elif chosen_route_name_violate is not None:
            if insert_idx_no_violate == 0:
                previous_coord = oncall_order.coord
            else:
                previous_coord = planned_orders[insert_idx_violate-1].coord
            next_coord = planned_orders[insert_idx_violate].coord
            duration_incr = min_duration_violate
            is_valid = 0
            insert_idx = insert_idx_violate
        return is_valid, duration_incr, previous_coord, next_coord, insert_idx

    def get_route_state(self):
        route_res = []
        route_meta_info_dict = self.event.route_meta_info_dict
        for route_name in route_meta_info_dict.keys():
            is_valid, duration_incr, previous_coord, next_coord, insert_idx = self.get_actions_on_route(route_name)
            if is_valid == 1:
                route_res.append((duration_incr, previous_coord, next_coord, route_name, insert_idx))
            elif is_valid == 0:
                route_res.append((duration_incr+NO_VALID_PENALTY_COEF, previous_coord, next_coord, route_name, insert_idx))
        route_res = sorted(route_res, key=lambda x: x[0])
        state = np.zeros(NUM_ROUTE*5)
        self.action_mask = np.zeros(NUM_ROUTE+1)
        self.action_mask[-1] = 1
        for i in range(min(len(route_res), NUM_ROUTE)):
            self.action_mask[i] = 1
            state[i:i+5] = ([route_res[i][0] / DURATION_NORM]
                             + [route_res[i][1].lat / LAT_NORM, route_res[i][1].lng / LONG_NORM]
                             + [route_res[i][2].lat / LAT_NORM, route_res[i][2].lng / LONG_NORM])
        self.action_to_route = route_res
        return state
    
    # information about other oncall orders that are closed to the current one
    def get_oncall_state(self):
        order_dist = []
        tick = self.running_env.tick
        oncall_orders = self.event.oncall_orders
        est_duration_predictor = self.event.estimated_duration_predictor
        for order in oncall_orders:
            val = est_duration_predictor.predict(tick, order.coord, self.cur_oncall_order.coord)
            order_dist.append((val, order.coord))
        order_dist = sorted(order_dist, key=lambda x: x[0])
        state = np.zeros(NUM_ONCALL_ORDERS*3)
        for i in range(min(len(order_dist), NUM_ONCALL_ORDERS)):
            state[i:i+3] = ([order_dist[i][0] / DURATION_NORM] 
                            + [order_dist[i][1].lat / LAT_NORM, order_dist[i][1].lng / LONG_NORM])
        return state
    
    # global information about all existing routes and their planned stops
    # max, min, median of all langtitudes and logtitudes
    # max, min, median of number of planned stops on all routes
    def get_global_state(self):
        global_state = np.zeros(9)
        if self.event:
            lat_list = []
            lng_list = []
            stops_list = []
            route_plan_dict = self.event.route_plan_dict
            for _, routes in route_plan_dict.items():
                stops_list.append(len(routes) / MAX_NUM_STOPS_PER_ROUTE)
                for order in routes:
                    lat_list.append(order.coord.lat / LAT_NORM)
                    lng_list.append(order.coord.lng / LONG_NORM)
            if len(lat_list) > 0:
                global_state[:3] = [np.max(lat_list), np.min(lat_list), np.median(lat_list)]
            if len(lng_list) > 0:
                global_state[3:6] = [np.max(lng_list), np.min(lng_list), np.median(lng_list)]
            if len(stops_list) > 0:
                global_state[6:] = [np.max(stops_list), np.min(stops_list), np.median(stops_list)]
        return global_state

    def get_state(self):
        global_state = self.get_global_state()
        route_state = self.get_route_state()
        oncall_state = self.get_oncall_state()
        self.state = np.concatenate([global_state, oncall_state, route_state, self.action_mask], axis=0)
        return self.state

    def get_delay_tick(self, delay_orders):
        delay_tick = 0
        for order, tick in delay_orders:
            delay_tick += (tick - order.close_time)
        return delay_tick

    def step(self, action):
        self.episode_steps += 1
        if type(action) is np.ndarray:
            action = action[0]
        if action == NUM_ROUTE:
            if len(self.action_to_route) <= 0: 
                env_action = [PostponeAction(order_id=self.cur_oncall_order.id)]
                duration = REWARD_NORM
            else:
                duration, _, _, route_name, insert_idx = self.action_to_route[0]
                env_action = [AllocateAction(order_id=self.cur_oncall_order.id, route_name=route_name, insert_index=insert_idx, in_segment_order=0)]
        else:
            duration, _, _, route_name, insert_idx = self.action_to_route[action]
            env_action = [AllocateAction(order_id=self.cur_oncall_order.id, route_name=route_name, insert_index=insert_idx, in_segment_order=0)]
        
        self.reward = - duration / REWARD_NORM
        self.metrics, self.event, self.done = self.running_env.step(env_action)
        while (not self.done and (self.event is None)):
            self.metrics, self.event, self.done = self.running_env.step([])
        
        if self.event:
            self.cur_oncall_order = self.event.oncall_orders[0]
            self.state = self.get_state()
            
            _total_delayed_tick = self.get_delay_tick(self.event.delayed_orders)
            _total_terminated_tick = self.get_delay_tick(self.event.delayed_orders)
            # self.reward -= 2.0*(_total_delayed_tick - self.total_delayed_tick) / REWARD_NORM
            # self.reward -= 4.0*(_total_terminated_tick - self.total_terminated_tick) / REWARD_NORM
            self.total_delayed_tick = _total_delayed_tick
            self.total_terminated_tick = _total_terminated_tick
            
            self.total_delayed_num = len(self.event.delayed_orders)
            self.total_terminated_num = len(self.event.terminated_orders)
            
        return self.state, self.reward, self.done, {}


    
if __name__ == "__main__":
    env = Routing_Environment()
    max_route_num = float("-inf")
    max_stops_per_route = float("-inf")
    for _ in range(100):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(0)
            if env.event:
                route_plan_dict = env.event.route_plan_dict
                max_route_num = max(max_route_num, len(route_plan_dict))
                for route_name, routes in route_plan_dict.items():
                    max_stops_per_route = max(max_stops_per_route, len(routes))
    
    print(max_route_num, max_stops_per_route)
