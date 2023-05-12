# DICP
## Quick Start
1. pytorch
  - clone pytorch (DICP_dev branch)
  - build pytorch
2. DICP
  - clone DICP (main branch)
  - cd DICP
  - for users
    - pip install .
  - for dev
    - export PYTHONPATH=$PWD:$PYTHONPATH
3. demo
  - TopsGraph
    - cd DICP/test/Tops/
    - python test_entry_point.py
    - python resnet_precison.py
    - python resnet_performance.py
  - AscendGraph
    - cd DICP/test/Ascend/
    - python test_resnet18.py