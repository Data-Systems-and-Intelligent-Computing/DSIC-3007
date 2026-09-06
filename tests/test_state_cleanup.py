from src.verify import verify_state_cleanup

def test_state_cleanup():
    verify_state_cleanup(1.8, 1.8, atol=1e-8)
