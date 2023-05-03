import numpy as np
import os
import segyio

def main():
    root_dir = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/origin_data/seis/mig_fill.sgy'
    # seis = np.load(os.path.join(root_dir, 'seistrain.npy'))
    seis = segyio.tools.cube(root_dir)
    print(f'mean is {seis.mean()}, std is {seis.std()}')

if __name__ == '__main__':
    main()