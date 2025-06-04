from mp import PerichMillerPopulation2019

x = PerichMillerPopulation2019(data_root="./data/processed")
x.process(raw_root="./data/root", n_cpus=2)
