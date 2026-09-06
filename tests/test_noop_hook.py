from src.verify import verify_noop_hook

def test_noop_hook_equivalence():
    verify_noop_hook(2.123456, 2.123456, atol=1e-8)
