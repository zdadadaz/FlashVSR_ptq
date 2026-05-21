import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

def test_nodes_has_w8a8_ptq_mode():
    import nodes
    import inspect
    source = inspect.getsource(nodes.init_pipeline)
    assert 'W8A8_PTQ' in source, "W8A8_PTQ not found in init_pipeline"
    print("test_nodes_has_w8a8_ptq_mode PASS")

def test_load_trt_engine_exists():
    import nodes
    assert hasattr(nodes, 'load_trt_engine'), "load_trt_engine not found"
    print("test_load_trt_engine_exists PASS")

def test_trt_engine_path_param():
    import nodes
    import inspect
    sig = inspect.signature(nodes.init_pipeline)
    assert 'trt_engine_path' in sig.parameters, "trt_engine_path not in init_pipeline sig"
    print("test_trt_engine_path_param PASS")