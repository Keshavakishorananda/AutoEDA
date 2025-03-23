import numpy as np
from typing import NewType
from enum import Enum
from abc import ABC, abstractmethod

Column = NewType('Column', str)


class FilterOperator(str, Enum):
    EQUAL = 'EQ'
    NOTEQUAL = 'NEQ'
    CONTAINS = 'CONTAINS'
    GT = 'GT'
    GE = 'GE'
    LT = 'LT'
    LE = 'LE'
    STARTS_WITH = 'STARTS_WITH'
    ENDS_WITH = 'ENDS_WITH'


class AggregationFunction(Enum):
    COUNT = 0
    SUM = 1
    MAX = 2
    MIN = 3
    MEAN = 4


class ActionType(Enum):
    BACK = 'BACK'
    FILTER = 'FILTER'
    GROUP = 'GROUP'


class AbstractAction(ABC):
    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    @abstractmethod
    def __repr__(self):
        return NotImplementedError


class BackAction(AbstractAction):
    def __init__(self):
        super().__init__(ActionType.BACK)

    def str_rep(self):
        return 'BACK'

    def __repr__(self):
        return self.str_rep()

    def __hash__(self):
        return hash(self.str_rep())

    def __eq__(self, other):
        return self.str_rep() == other.str_rep()


class FilterAction(AbstractAction):
    def __init__(self, filtered_column: Column, filter_operator: FilterOperator, filter_term: str):
        super().__init__(ActionType.FILTER)
        self.filtered_column = filtered_column
        self.filter_operator = filter_operator
        self.filter_term = filter_term

    def str_rep(self):
        return f'FILTER {self.filtered_column} {self.filter_operator}  {self.filter_term}'

    def __repr__(self):
        return self.str_rep()

    def __hash__(self):
        return hash(self.str_rep())

    def __eq__(self, other):
        return self.str_rep() == other.str_rep()
        # return (self.filtered_column == other.filtered_column) and (self.filter_operator == other.filter_operator) and(self.filter_term == other.filter_term)


class GroupAction(AbstractAction):
    def __init__(self, grouped_column: Column, aggregated_column: Column, aggregation_function: AggregationFunction):
        super().__init__(ActionType.GROUP)
        self.grouped_column = grouped_column
        self.aggregated_column = aggregated_column
        self.aggregation_function = aggregation_function

    def str_rep(self):
        return f'GROUP {self.grouped_column} AGGREGATE {self.aggregation_function} {self.aggregated_column}'

    def __repr__(self):
        return self.str_rep()

    def __hash__(self):
        return hash(self.str_rep())

    def __eq__(self, other):
        return self.str_rep() == other.str_rep()

def PrecisionMetric(actions_lst):
    pass

if __name__ == "__main__":
    actions_lst = [
        BackAction(),
        FilterAction(),
        GroupAction(),
    ]
    PrecisionMetric(actions_lst)