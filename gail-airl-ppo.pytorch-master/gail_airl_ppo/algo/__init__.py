from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from ....AIRL import AIRL

ALGOS = {
    'gail': GAIL,
    'airl': AIRL
}
