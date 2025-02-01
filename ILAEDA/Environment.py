import numpy as np
from DatasetProp import DatasetProp
import gymnasium as gym
from collections import namedtuple
import pandas as pd
import operator
import logging
from scipy.stats import entropy
from collections import Counter

class CounterWithoutNanKeys(Counter):
    def __init__(self, iterable):
        self.non_nan_iterable = [elem for elem in iterable if isinstance(elem, str) or not np.isnan(elem)]
        super().__init__(self.non_nan_iterable)

logger = logging.getLogger(__name__)

MAX_PREV_STATES = 3

MAX_EPISODES_STEPS = 12

ACTIONS = ['back', 'filter', 'group', 'stop']

OPERATORS = {
    0 : 'EQ',
    1 : 'NEQ',
    2 : 'GT',
    3 : 'GE',
    4 : 'LT',
    5 : 'LE',
    6 : 'CONTAINS',
    7 : 'STARTSWITH',
    8 : 'ENDSWITH',
}

OPERATOR_MAP = {
    'EQ': operator.eq,
    'NEQ': operator.ne,
    'GT': operator.gt,
    'GE': operator.ge,
    'LT': operator.lt,
    'LE': operator.le,
}

def hack_min(pd_series):
    return np.min(pd_series.dropna())


def hack_max(pd_series):
    return np.max(pd_series.dropna())

AGGREGATION_FUNCTIONS = {
    0 : len,
    1 : np.sum,
    2 : hack_min,
    3 : hack_max,
    4 : np.mean,
}

newtuple = namedtuple('newtuple', ['Filtering', 'Groupings', 'Aggregations'])

class State_store():
    
    def __init__(self):
        self.statetuple = namedtuple('statetuple', ['Filtering', 'Groupings', 'Aggregations'])
        self.state_tuple = self.statetuple([], [], [])

    def create_empty_state_tuple(self):
        self.state_tuple = self.statetuple([], [], [])

        return self.state_tuple

    def add_filtering(self, filtering):
        self.state_tuple.Filtering.append(filtering)
        return self.state_tuple

    def add_grouping(self, grouping):
        self.state_tuple.Groupings.append(grouping)
        return self.state_tuple

    def add_aggregation(self, aggregation):
        self.state_tuple.Aggregations.append(aggregation)
        return self.state_tuple


Empty_state_class = State_store()
Empty_state = Empty_state_class.create_empty_state_tuple()

class Environment():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self.dataset_prop = DatasetProp(self.dataset_path)
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()
        
    def reset(self):
        # print("Resetting environment")
        self.step_num = 0
        self.state_stack = [Empty_state]
        self.state_history = [Empty_state]
        self.action_history = []

        self.obs_history = []
        obs_vector = self.get_observation_vector()
        self.obs_history.append(obs_vector)

        self.display_history = [self.get_display_vector(self.state_history[0])]

        return obs_vector

    def step(self, action): 
        # print("Taking step")
        action = self.preprocess_action(action)
        self.action_history.append(action)
        self.take_action(action)

        observation_vector = self.get_observation_vector()
        self.obs_history.append(observation_vector)
        self.display_history.append(self.get_display_vector(self.state_history[-1]))

        self.step_num += 1
        terminated = True if action['type'] == 'stop' else False
        truncated = True if self.step_num >= MAX_EPISODES_STEPS else False
        done = terminated or truncated
        done = True if (self.step_num >= MAX_EPISODES_STEPS or action['type'] == 'stop') else False

        return observation_vector, 0, terminated, truncated, {}

    # For Init function
    def get_observation_space(self):
        
        nof_colums = len(self.dataset_prop.Keys)
        low_datalayer = np.zeros(nof_colums*3)
        high_datalayer = np.ones(nof_colums*3)

        # Whether an attribute is grouped or aggregated.
        # -1: Neither grouped nor aggregated
        # v \in [0, 1]: Aggregated, where v is normalized entropy of the
        #               of the attribute in the aggregated dataframe
        # 2: Grouped
        low_granularitylayer = np.full(nof_colums, -1)
        high_granularitylayer = np.full(nof_colums, 2)

        low_globallayer = np.zeros(3)
        high_globallayer = np.ones(3)

        low_single = np.concatenate([low_datalayer, low_granularitylayer, low_globallayer])
        high_single = np.concatenate([high_datalayer, high_granularitylayer, high_globallayer])

        low = np.tile(low_single, MAX_PREV_STATES)
        high = np.tile(high_single, MAX_PREV_STATES)

        return gym.spaces.Box(low, high, shape=(len(low), ))

    def get_action_space(self):
        
        num_actions = len(ACTIONS)
        nof_colums = len(self.dataset_prop.Keys)
        num_operators = len(OPERATORS.keys())
        num_bins = len(self.dataset_prop.bins) - 1
        num_agg_options = nof_colums + 1
        num_agg_functions = len(AGGREGATION_FUNCTIONS.keys())

        return gym.spaces.MultiDiscrete((num_actions, nof_colums, num_operators, num_bins, num_agg_options, num_agg_functions))

    # For Reset function
    def get_observation_vector(self):
        display_dict = self.get_display_vector(self.state_history[-1])

        last_display_vector = []
        for d in display_dict['datalayer'].values():
            last_display_vector += [d['unique'], d['null'], d['entropy']]

        if display_dict['global_layer'] is None:
            last_display_vector += [-1 for _ in range(len(self.dataset_prop.Keys))] + [0 for _ in range(3)]
        else:
            for k in self.dataset_prop.Keys:
                if k in display_dict['global_layer']['agg_attr'].keys():
                    last_display_vector.append(display_dict['global_layer']['agg_attr'][k])
                elif k in display_dict['global_layer']['group_attr']:
                    last_display_vector.append(2)
                else:
                    last_display_vector.append(-1)

            last_display_vector += [display_dict['global_layer']['inverse_ngroups'], display_dict['global_layer']['site_std'], display_dict['global_layer']['inverse_size_mean']]


        if not self.obs_history:
            padding = np.zeros(2 * len(last_display_vector))
            return np.concatenate([padding, last_display_vector])

        last_obs_vector = self.obs_history[-1]
        obs_vector = np.concatenate([last_obs_vector[-2 * len(last_display_vector):], last_display_vector])

        return np.array(obs_vector)


    def get_display_vector(self, state):
        filtered_df = self.get_filtered_df(state)
        grouped_df, aggregated_df = self.get_grouped_and_aggregated_df(state, filtered_df)
        display_dict = self.df_to_vector(filtered_df, grouped_df, aggregated_df)

        return display_dict

    def df_to_vector(self, filtered_df, grouped_df, aggregated_df):

        display = {}
        bins = B = 20

        # Datalayer
        display_datalayer = {}
        for column in self.dataset_prop.Keys:
            if len(filtered_df[column]) == 0:
                display_datalayer[column] = {}
                display_datalayer[column]['unique'] = 0
                display_datalayer[column]['null'] = 0
                display_datalayer[column]['entropy'] = 0
            else:
                display_datalayer[column] = {}

                column_na_value_counts = CounterWithoutNanKeys(filtered_df[column])
                column_na_value_counts_values = column_na_value_counts.values()
                cna_size = sum(column_na_value_counts_values)

                n = len(filtered_df[column]) - cna_size  # nof null values
                u = len(column_na_value_counts.keys())  # nof unique non-null values
                u_n = u / cna_size if u != 0 else 0

                if column not in self.dataset_prop.numeric_keys:
                    h = entropy(list(column_na_value_counts_values))
                    h = h / np.log(cna_size) if cna_size > 1 else 0.0
                else:
                    h = entropy(np.histogram(column_na_value_counts.non_nan_iterable, bins=B)[0]) / np.log(B) if cna_size > 1 else 0.0

                display_datalayer[column]['unique'] = u_n
                display_datalayer[column]['null'] = n / len(filtered_df[column])
                display_datalayer[column]['entropy'] = h

        display['datalayer'] = display_datalayer

        # Globallayer
        if grouped_df is None or aggregated_df is None:
            display['global_layer'] = None
            return display

        B = 20
        groups_num = grouped_df.ngroups
        if groups_num == 0:
            site_std = 0.0
            inverse_size_mean = 0
            inverse_ngroups = 0
        else:
            sizes = grouped_df.size()
            sizes_sum = sizes.sum()
            nsizes = sizes / sizes_sum
            site_std = nsizes.std(ddof=0)
            sizes_mean = sizes.mean()
            inverse_size_mean = 1 / sizes_mean
            if sizes_sum > 0:
                inverse_ngroups = 1 / groups_num
            else:
                inverse_ngroups = 0
                inverse_size_mean = 0

        group_keys = grouped_df.keys
        agg_keys = list(aggregated_df.keys())
        agg_nve_dict = {}
        if agg_keys is not None:
            for ak in agg_keys:
                column = aggregated_df[ak]
                column_na = column.dropna()
                cna_size = len(column_na)
                if cna_size <= 1:
                    agg_nve_dict[ak] = 0.0
                elif aggregated_df[ak].dtype == 'O' or str(aggregated_df[ak].dtype) == 'category':
                    h = entropy(column_na.value_counts().values)
                    agg_nve_dict[ak] = h / np.log(cna_size)
                else:
                    agg_nve_dict[ak] = entropy(np.histogram(
                        column_na, bins=B)[0]) / np.log(B)

        display_global = {}
        display_global['group_attr'] = group_keys
        display_global['agg_attr'] = agg_nve_dict
        display_global['inverse_ngroups'] = inverse_ngroups
        display_global['site_std'] = site_std
        display_global['inverse_size_mean'] = inverse_size_mean
        display['global_layer'] = display_global

        return display

    def get_filtered_df(self, state):
        filters = state.Filtering
        dataset_df = self.dataset_prop.df

        if len(filters) == 0:
            return dataset_df

        df = dataset_df.copy()

        for filter in filters:
            field = filter[0]
            opr = filter[2]
            value = filter[1]

            if opr in ['EQ', 'NEQ', 'GT', 'GE', 'LT', 'LE']:
                opr = OPERATOR_MAP[opr]
                try:
                    if pd.api.types.is_numeric_dtype(df[field]) and value == '<UNK>':
                        value = np.nan
                    value = float(value) if str(df[field].dtype) not in [
                        'object', 'category'] and value != '<UNK>' else value
                    df = df[opr(df[field], value)]
                except:
                    return df.truncate(after=-1)
            else:
                try:
                    if opr == 'CONTAINS':
                        if df[field].dtype == 'O' or str(df[field].dtype) == 'category':
                            df = df[df[field].str.contains(
                                value, na=False, regex=False)]
                        elif df[field].dtype == 'f8' or df[field].dtype == 'u4' or df[field].dtype == 'int64':
                            df = df[df[field].astype(str).str.contains(
                                str(value), na=False, regex=False)]
                        else:
                            logger.warning(
                                f"Filter on column {field} with operator Contains and value {value} is emtpy")
                            raise NotImplementedError

                    elif opr == 'STARTSWITH':
                        if df[field].dtype == 'O' or str(df[field].dtype) == 'category':
                            df = df[df[field].str.startswith(value, na=False)]
                        elif df[field].dtype == 'f8' or df[field].dtype == 'u4' or df[field].dtype == 'int64':
                            df = df[df[field].astype(str).str.startswith(
                                str(value), na=False)]
                        else:
                            logger.warning(
                                f"Filter on column {field} with operator Contains and value {value} is emtpy")
                            raise NotImplementedError

                    elif opr == 'ENDSWITH':
                        if df[field].dtype == 'O' or str(df[field].dtype) == 'category':
                            df = df[df[field].str.endswith(value, na=False)]
                        elif df[field].dtype == 'f8' or df[field].dtype == 'u4' or df[field].dtype == 'int64':
                            df = df[df[field].astype(str).str.endswith(
                                str(value), na=False)]
                        else:
                            logger.warning(
                                f"Filter on column {field} with operator Contains and value {value} is emtpy")
                            raise NotImplementedError
                    else:
                        logger.warning(
                            f"Filter on column {field} with operator {opr} and value {value} raised NotImplementedError and will be emtpy")
                        raise NotImplementedError

                except NotImplementedError:
                    return df.truncate(after=-1)
        return df
    
    def get_grouped_and_aggregated_df(self, state, filtered_df):
        groupings = state.Groupings
        aggregations = state.Aggregations
        df = filtered_df

        if not groupings:
            return None, None

        df_gb = df.groupby(list(groupings), observed=True)

        agg_dict = {}
        for agg in aggregations:
            agg_dict[agg[0]] = agg[1]

        try:
            agg_df = df_gb.agg(agg_dict)
        except:
            return None, None
        return df_gb, agg_df
    
    # For Step function
    def preprocess_action(self, action):

        ret_action = {
            "type": None,
            "col": None,
            "operator": None,
            "filter_term": None,
            "agg_option": None,
            "agg_func": None
        }

        filter_term_bin = action[3]

        action = np.round(action).astype(int)

        ret_action['type'] = ACTIONS[action[0]]

        if ret_action['type'] in ['back', 'stop']:
            return ret_action

        ret_action['col'] = self.dataset_prop.Keys[action[1]]

        if ret_action['type'] == 'filter':
            ret_action['operator'] = OPERATORS[action[2]]
            try:
                ret_action['filter_term'] = self.get_filter_term(ret_action['col'], action[3])
            except:
                ret_action['filter_term'] = '<UNK>'
            return ret_action

        if ret_action['type'] == 'group':
            if action[4] == 0:
                # Check if aggregation is over no column
                ret_action['agg_option'] = 'number'
                ret_action['agg_func'] = len
            else:
                # Aggregation column index is action[4] - 1
                ret_action['agg_option'] = self.dataset_prop.Keys[action[4] - 1]
                ret_action['agg_func'] = AGGREGATION_FUNCTIONS[action[5]]
            return ret_action

    def take_action(self, action):
        if action['type'] == 'back':
            if len(self.state_stack) > 1:
                self.state_stack.pop()
                new_state = self.state_stack[-1]
                self.state_history.append(new_state)
            else:
                new_state = self.state_stack[-1]
                self.state_history.append(new_state)

        elif action['type'] == 'stop':
            self.state_stack = [Empty_state]
            self.state_history.append(Empty_state)

        elif action['type'] == 'filter':
            filtering_tuple = (action['col'], action['filter_term'], action['operator'])
            temp_state = self.state_history[-1]
            if filtering_tuple not in temp_state.Filtering:
                new_state = newtuple(temp_state.Filtering + [filtering_tuple], temp_state.Groupings, temp_state.Aggregations)
            else:
                new_state = temp_state
            self.state_stack.append(new_state)
            self.state_history.append(new_state)

        elif action['type'] == 'group':
            temp_state = self.state_history[-1]
            group_int = 0
            aggr_int = 0
            if action['col'] not in temp_state.Groupings:
                group_int = 1
            agg_tuple = (action['agg_option'], action['agg_func'])
            if agg_tuple not in temp_state.Aggregations:
                aggr_int = 1

            if group_int == 0 and aggr_int == 0:
                new_state = newtuple(temp_state.Filtering, temp_state.Groupings, temp_state.Aggregations)
                
            if group_int == 1 and aggr_int == 0:
                new_state = newtuple(temp_state.Filtering, temp_state.Groupings + [action['col']], temp_state.Aggregations)

            if group_int == 0 and aggr_int == 1:
                new_state = newtuple(temp_state.Filtering, temp_state.Groupings, temp_state.Aggregations + [agg_tuple])

            if group_int == 1 and aggr_int == 1:
                new_state = newtuple(temp_state.Filtering, temp_state.Groupings + [action['col']], temp_state.Aggregations + [agg_tuple])

            self.state_history.append(new_state)
            self.state_stack.append(new_state)

    def get_filter_term(self, col, bin):
        last_state = self.state_history[-1]
        df = self.get_filtered_df(last_state)

        if len(df) == 0:
            return '<UNK>'

        # term_pos is a position between 0 and 1 used to do nearest neighbour search in
        # a frequency sorted list of filter terms from the queried col of the df
        term_pos = self.dataset_prop.bins[bin] + (
            self.dataset_prop.bins[bin + 1] - self.dataset_prop.bins[bin]) * np.random.random()

        return self.get_nearest_neighbour_filter_term(df, col, term_pos)
    
    def get_nearest_neighbour_filter_term(self, df, col, term_pos):
        filter_terms = df[col].dropna().tolist()

        freq_dict = {}
        for term in filter_terms:
            if term not in freq_dict:
                freq_dict[term] = 0
            freq_dict[term] += 1

        freq_tups = sorted(list(freq_dict.items()), key=lambda kv: kv[1])
        freq_keys = np.array([kv[0] for kv in freq_tups])
        freq_vals = np.array([kv[1] for kv in freq_tups])
        freq_vals = freq_vals / np.sum(freq_vals)
        cumulative_freqs = np.cumsum(freq_vals)

        return freq_keys[np.abs(cumulative_freqs - term_pos).argmin()]
    
    # For rewards
    def repeat_penalty(self):
        if self.action_history is None or len(self.action_history) < 2:
            return 0
        
        if self.state_history[-2] == Empty_state and self.action_history[-1]['type'] == 'back':
            return -1.0
        
        if self.action_history[-1] != self.action_history[-2]:
            # Penalty for alternating BACK repeats
            if self.action_history[-1]['type'] == 'back':
                alt_repeat_len = self.get_alternate_back_repeating_length()
                if alt_repeat_len > 1:
                    return -1.0 * alt_repeat_len
                else:
                    return 0.0
            else:
                return 0.0
            

        if self.action_history[-1] == self.action_history[-2]:
            if self.action_history[-1]['type'] in ['filter', 'group']:
                return -1.0
            else:
                return 0.0
        
        else:
            return 0
        
    
    def get_alternate_back_repeating_length(self):
        l = 0
        for i, action in enumerate(reversed(self.action_history)):
            if i % 2 == 1:
                if action['type'] != 'back':
                    l += 1
                else:
                    break
            elif i % 2 == 0:
                if action['type'] != 'back':
                    break
        return l




class MultidatasetEnvironment(gym.Env):
    def __init__(self, dataset_paths, max_steps = MAX_EPISODES_STEPS):
        super(MultidatasetEnvironment, self).__init__()

        self.envs = [Environment(path) for path in dataset_paths]
        self.nums_envs = len(self.envs)
        self.current_env = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_env = (self.current_env + 1) % self.nums_envs
        return self.envs[self.current_env].reset(), {}
    
    def step(self, action):
        return self.envs[self.current_env].step(action)
    
    def __getattr__(self, name):
        return getattr(self.envs[self.current_env], name)
