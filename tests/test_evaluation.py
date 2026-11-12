import numpy as np
from src.evaluation import false_discovery_rate, choose_threshold, evaluate

def test_fdr_definition():
    assert false_discovery_rate([0,0,1,1],[1,0,1,0]) == .5

def test_threshold_is_probability():
    t=choose_threshold(np.array([0,0,0,1,1,1]),np.array([.1,.2,.3,.6,.7,.8]),.5)
    assert 0 <= t <= 1

def test_confusion_matrix_and_fdr():
    m=evaluate([0,0,1,1],[.1,.8,.7,.2],.5)
    assert m["confusion_matrix"] == {"tn":1,"fp":1,"fn":1,"tp":1}
    assert m["false_discovery_rate"] == .5
