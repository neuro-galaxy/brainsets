import h5py
import numpy as np
import tensorstore as ts
from temporaldata import Data, Interval, RegularTimeSeries
from zapbench.constants import CONDITION_OFFSETS, TEST_FRACTION, VAL_FRACTION

from brainsets import serialize_fn_map
from brainsets.descriptions import BrainsetDescription

BASE_URL = "vast/projects/dyer1/lab/user/alex/20240930/traces/zapbench.h5"


spec = {
    "driver": "zarr3",
    "kvstore": {
        "driver": "file",
        "path": f"{BASE_URL}/traces",
    },
}
raw = ts.open(spec).result()

traces = RegularTimeSeries(raw=raw.T, sampling_rate=1.0, domain="auto")

brainset_description = BrainsetDescription(
    id="zapbench",
    origin_version="",
    derived_version="1.0.0",
    source="",
    description="",
)

nb_neurons = raw.shape[0]
condition = np.zeros(nb_neurons)
starts, ends = CONDITION_OFFSETS[:-1], CONDITION_OFFSETS[1:]
for i, (start, end) in enumerate(zip(starts, ends)):
    condition[start:end] = i
data = Data(
    brainset=brainset_description,
    traces=traces,
    condition=condition,
    domain="auto",
)

new_offset = np.array(CONDITION_OFFSETS[:2] + CONDITION_OFFSETS[4:])
delta = new_offset[1:] - new_offset[:-1]
# Train
train_start = new_offset[:-1]
train_end = new_offset[:-1] + (1 - VAL_FRACTION - TEST_FRACTION) * delta
train_interval = Interval(start=train_start, end=train_end)
# Val
val_start = train_end
val_end = new_offset[:-1] + (1 - TEST_FRACTION) * delta
val_interval = Interval(start=val_start, end=val_end)
# Test
test_start = val_end
test_end = new_offset[1:]
test_interval = Interval(start=test_start, end=test_end)

data.set_train_domain(train_interval)
data.set_valid_domain(val_interval)
data.set_test_domain(test_interval)

store_path = f"{BASE_URL}/zapbench.h5"
with h5py.File(store_path, "w") as file:
    data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
