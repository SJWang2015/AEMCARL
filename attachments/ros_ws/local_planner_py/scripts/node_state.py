#! /usr/bin/env python
from enum import Enum


class NodeState(Enum):
    IDLE = 0
    RUNNING = 1
    PAUSE = 2
    SUCCESS = 3
    FAILURE = 4