from envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.gatcarl import GATCARL
from crowd_nav.policy.comcarl import COMCARL
from crowd_nav.policy.actenvcarl import ACTENVCARL

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['gatcarl'] = GATCARL
policy_factory['comcarl'] = COMCARL
policy_factory['actenvcarl'] = ACTENVCARL
