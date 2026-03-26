import h5py

# f = h5py.File('/workspace/oat_casa/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/mg/2024-05-09-09-32-19/demo_gentex_im128_randcams.hdf5', 'r')
f = h5py.File('/workspace/oat_casa/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/mg/demo_gentex_im128_randcams.hdf5', 'r')
# or wherever the file lands
print(list(f['data'].keys()))
demo = f['data/demo_0']
print('actions:', demo['actions'].shape, demo['actions'].dtype)
for k in demo['obs']:
    print(f'obs/{k}:', demo['obs'][k].shape, demo['obs'][k].dtype)
f.close()