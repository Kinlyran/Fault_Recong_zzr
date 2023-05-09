import numpy as np
import os
import segyio

def main():
    # root_dir = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/precessed'
    # seis = np.load(os.path.join(root_dir, 'seisval.npy'))
    seis = segyio.tools.cube('/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/origin_data/seis/mig_fill.sgy')
    print(f'mean is {seis.mean()}, std is {seis.std()}')

if __name__ == '__main__':
    main()