import os
import sys
import tempfile

# Ensure repo root is on sys.path so `import src...` works when running pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.metrics_collector import (
    Collector,
    Session,
    CounterInstrument,
    HistogramInstrument,
    ValueType,
)


def test_counter_and_export(tmp_path):
    c = Collector()
    # register timeseries with attribute key 'route'
    c.register_timeseries("requests", registered_attribute_keys=["route"], max_count=10)

    sess = Session(c)
    counter = CounterInstrument("requests_counter", binding_attributes={"route": "/home"})
    sess.add_instrument(counter, "requests")

    # add datapoints matching the binding
    for _ in range(3):
        c.add_datapoint("requests", ValueType.INT, 1, attributes={"route": "/home"})

    assert counter.total == 3.0

    export = sess.to_dict()
    assert "start_time" in export
    assert isinstance(export["instruments"], list)
    # instrument export should include our total
    inst_export = export["instruments"][0]["instrument"]
    assert inst_export["type"] == "CounterInstrument"
    assert inst_export["total"] == 3.0


def test_eviction_and_removal_notifications():
    c = Collector()
    # keep only 2 points so that eviction happens
    c.register_timeseries("latency", registered_attribute_keys=["route"], max_count=2)

    sess = Session(c)
    hist = HistogramInstrument("latency_hist", binding_attributes={"route": "/api"})
    sess.add_instrument(hist, "latency")

    # add three values: 10, 20, 30
    c.add_datapoint("latency", ValueType.FLOAT, 10.0, attributes={"route": "/api"})
    c.add_datapoint("latency", ValueType.FLOAT, 20.0, attributes={"route": "/api"})
    c.add_datapoint("latency", ValueType.FLOAT, 30.0, attributes={"route": "/api"})

    # After eviction, histogram should have the last two values: 20 + 30 = 50
    assert hist.count == 2
    assert abs(hist.sum - 50.0) < 1e-6


if __name__ == '__main__':
    # quick local run
    test_counter_and_export(tempfile.gettempdir())
    test_eviction_and_removal_notifications()
