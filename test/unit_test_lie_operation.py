import numpy as np
import pytest
from controller.gmpc_se3 import *
from manifpy import SE3, SE3Tangent
my_twist = np.random.rand(6) # [w, v]
print(my_twist)



def test_adjoint():
    # in manifpy, the adjoint of se3 is defined as:
    # [w^, v^
    # 0,  w^]
    # in the controller, the adjoint of se3 is defined as:
    # [w^, 0
    # v^, w^]

    SE3_Tang_manif_adj = SE3Tangent(np.hstack([my_twist[3:6], my_twist[0:3]])).smallAdj()
    SE3_Tang_adj = adjoint(my_twist)

    assert SE3_Tang_adj + np.transpose(SE3_Tang_manif_adj) == pytest.approx(0, abs=1e-6)

def test_coadjoint():
    # in controller, the coadjoint of se3 is defined as:
    # [-w^, -v^
    # 0, -w^]
    SE3_Tang_coadj = coadjoint(my_twist)
    SE3_Tang_manif_adj = SE3Tangent(np.hstack([my_twist[3:6], my_twist[0:3]])).smallAdj()
    assert SE3_Tang_coadj + SE3_Tang_manif_adj == pytest.approx(0, abs=1e-6)