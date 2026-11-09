from sentinelai.api.limits import WINDOW_S, SlidingWindowLimiter


def test_allows_up_to_limit_then_refuses():
    limiter = SlidingWindowLimiter(2)
    assert limiter.check("a", now=0.0) == (True, False)
    assert limiter.check("a", now=1.0) == (True, False)
    assert limiter.check("a", now=2.0) == (False, True)


def test_only_first_refusal_of_a_burst_is_flagged():
    limiter = SlidingWindowLimiter(1)
    limiter.check("a", now=0.0)
    assert limiter.check("a", now=1.0) == (False, True)
    assert limiter.check("a", now=2.0) == (False, False)
    assert limiter.check("a", now=WINDOW_S + 1) == (True, False)
    assert limiter.check("a", now=WINDOW_S + 2) == (False, True)


def test_clients_are_independent():
    limiter = SlidingWindowLimiter(1)
    assert limiter.check("a", now=0.0)[0]
    assert limiter.check("b", now=0.0)[0]
    assert not limiter.check("a", now=1.0)[0]
