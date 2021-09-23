from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.comcarl import COMCARL
from crowd_nav.policy.gipcarl import GIPCARL
from crowd_nav.policy.actcarl import ACTCARL
from crowd_nav.policy.actenvcarl import ACTENVCARL
from crowd_nav.policy.actfcarl import ACTFCARL


policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['comcarl'] = COMCARL
policy_factory['gipcarl'] = GIPCARL
policy_factory['actcarl'] = ACTCARL
policy_factory['actenvcarl'] = ACTENVCARL
policy_factory['actfcarl'] = ACTFCARL
