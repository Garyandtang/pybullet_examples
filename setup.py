from setuptools import setup

setup(name='safe_control_gym',
      version='2.0.0',
      install_requires=[
          'matplotlib',
          'Pillow',
          'munch',
          'pyyaml',
          'imageio',
          'dict-deep',
          'scikit-optimize',
          'gymnasium',
          'torch',
          'gpytorch',
          'tensorboard',
          'casadi',
          'pybullet',
          'cvxpy',
          'pytope',
          'Mosek',
          'termcolor',
      ],
      )
