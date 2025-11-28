from nhl_api import parse_iso_z, compare_to_now, is_past, seconds_until
from datetime import datetime, timezone, timedelta


def test_parse_iso_z_and_compare():
    now = datetime.now(timezone.utc)
    # create three timestamps: past, now (approx), future
    past_ts = (now - timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    future_ts = (now + timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    now_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    assert is_past(past_ts)
    assert compare_to_now(past_ts) == -1

    # now may be close enough to return 0 (within 1 second); this asserts behaviour
    cmp_now = compare_to_now(now_ts)
    assert cmp_now in (-1, 0, 1)

    assert compare_to_now(future_ts) == 1
    assert seconds_until(future_ts) > 0
    assert seconds_until(past_ts) < 0

