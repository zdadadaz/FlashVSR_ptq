import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

def test_flashvsr_full_has_model_fn_trt():
    from src.pipelines.flashvsr_full import FlashVSRFullPipeline
    assert hasattr(FlashVSRFullPipeline, 'model_fn_trt'), "model_fn_trt not found"
    print("test_flashvsr_full_has_model_fn_trt PASS")

def test_flashvsr_full_has_trt_engine_property():
    from src.pipelines.flashvsr_full import FlashVSRFullPipeline
    pipe = FlashVSRFullPipeline.__new__(FlashVSRFullPipeline)
    pipe.trt_engine_ = None
    assert hasattr(pipe, 'trt_engine'), "getter missing"
    assert hasattr(type(pipe), 'trt_engine') and isinstance(type(pipe).trt_engine, property)
    print("test_flashvsr_full_has_trt_engine_property PASS")

def test_trt_engine_setter_and_getter():
    from src.pipelines.flashvsr_full import FlashVSRFullPipeline
    pipe = FlashVSRFullPipeline.__new__(FlashVSRFullPipeline)
    pipe.trt_engine_ = None
    class FakeEngine:
        def __call__(self, x, t, ctx): return x
    fake = FakeEngine()
    pipe.trt_engine = fake
    assert pipe.trt_engine is fake
    print("test_trt_engine_setter_and_getter PASS")