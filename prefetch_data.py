from chronos_esm import data
try:
    print("prefetching data...")
    data.load_initial_conditions(nz=15)
    print("Success!")
except Exception as e:
    print(e)
