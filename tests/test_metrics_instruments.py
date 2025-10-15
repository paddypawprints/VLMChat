import os
import sys

# Ensure repo root is on sys.path so `import src...` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.metrics_collector import (
    Collector,
    Session,
    CountInstrument,
    AverageInstrument,
    HistogramByAttributeInstrument,
    ValueType,
)


def test_count_instrument_basic():
    c = Collector()
    c.register_timeseries("events", registered_attribute_keys=["type"], max_count=10)

    s = Session(c)
    inst = CountInstrument("event_count", binding_attributes={"type": "click"})
    s.add_instrument(inst, "events")

    c.add_datapoint("events", ValueType.INT, 1, attributes={"type": "click"})
    c.add_datapoint("events", ValueType.INT, 1, attributes={"type": "view"})
    c.add_datapoint("events", ValueType.INT, 1, attributes={"type": "click"})

    assert inst.count == 2


def test_average_instrument_basic():
    c = Collector()
    c.register_timeseries("values", registered_attribute_keys=["k"], max_count=10)

    s = Session(c)
    inst = AverageInstrument("avgval", binding_attributes={"k": "x"})
    s.add_instrument(inst, "values")

    c.add_datapoint("values", ValueType.FLOAT, 10.0, attributes={"k": "x"})
    c.add_datapoint("values", ValueType.FLOAT, 20.0, attributes={"k": "x"})
    c.add_datapoint("values", ValueType.FLOAT, 30.0, attributes={"k": "y"})

    assert abs(inst.average() - 15.0) < 1e-6


def test_histogram_by_attribute_basic():
    c = Collector()
    c.register_timeseries("measure", registered_attribute_keys=["region"], max_count=10)

    s = Session(c)
    inst = HistogramByAttributeInstrument("hist_by_region", bucket_key="region")
    s.add_instrument(inst, "measure")

    c.add_datapoint("measure", ValueType.INT, 5, attributes={"region": "us"})
    c.add_datapoint("measure", ValueType.INT, 7, attributes={"region": "eu"})
    c.add_datapoint("measure", ValueType.INT, 3, attributes={"region": "us"})

    b = inst.buckets
    assert "us" in b and "eu" in b
    assert b["us"]["sum"] == 8.0 and b["us"]["count"] == 2
    assert b["eu"]["sum"] == 7.0 and b["eu"]["count"] == 1
