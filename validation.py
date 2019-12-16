import importlib
import argparse
import pdb
from PIL import Image
import os
import numpy as np

import torch
import tqdm

from datasets.aligned_dataset import AlignedDataset
from datasets.data_loader import MyDataLoader
from models import base_solver, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default=None)
    parser.add_argument('--state_path', type=str, default=None)

    args = parser.parse_args()

    if args.state_path is not None:
        solver = base_solver.get_solver_from_solver_state(args.state_path)
        solver_conf = solver.conf['solver_conf']
        loader_conf = solver.conf['loader_conf']

    else:
        ## Load parameters ##
        conf = importlib.import_module(args.conf_path).conf
        solver_conf = conf['solver_conf']
        loader_conf = conf['loader_conf']
        solver_path = solver_conf['solver_path']
        solver = importlib.import_module(solver_path).get_solver(conf)
        if solver_conf['load_state']:
            solver.load_solver_state(torch.load(solver_conf['solver_state_path']))


    val_dataset = AlignedDataset(loader_conf, phase='val')

    val_loader = MyDataLoader(val_dataset, loader_conf, phase='val').get_loader()

    tq = tqdm.tqdm(val_loader)
    tq.set_description('{} | Epoch: {} | Step: {}'.format('val', 0, 0))

    psnr = []
    clean_paths = []
    cnt = 0
    save_dir = './temp'
    for idx, batch in enumerate(tq):
        real_A = batch['A'].numpy()[:, 0:3, :, :]
        real_B = batch['B'].numpy()
        cur_state = solver.process_batch(batch, 'val')
        fake_B = cur_state['else|fake_B'].numpy()

        path = batch['path']

        # pdb.set_trace()
        for i in range(real_B.shape[0]):
            sample_real_B, sample_fake_B, sample_real_A = map(lambda x:
                                                              ((x+1)/2.0*255.0).astype(np.uint8),
                                                              [real_B[i], fake_B[i], real_A[i]])

            sample_real_A = np.transpose(sample_real_A, [1, 2, 0])
            sample_real_B = np.transpose(sample_real_B, [1, 2, 0])
            sample_fake_B = np.transpose(sample_fake_B, [1, 2, 0])

            cur_psnr = metrics.PSNR(sample_real_B, sample_fake_B)
            psnr.append(cur_psnr)

            if cur_psnr < 40:
                Image.fromarray(sample_real_A).save(os.path.join(save_dir, str(cnt)+'_real_A_'+str(psnr[-1])+'.jpg'))
                Image.fromarray(sample_real_B).save(os.path.join(save_dir, str(cnt)+'_real_B_'+str(psnr[-1])+'.jpg'))
                Image.fromarray(sample_fake_B).save(os.path.join(save_dir, str(cnt)+'_fake_B_'+str(psnr[-1])+'.jpg'))
                cnt += 1
            else:
                clean_paths.append(path[i]+'\n')
        # pdb.set_trace()
    # with open('./val_clean.txt', 'w') as f:
    #     f.writelines(clean_paths)

    print(psnr)
    print(sum(psnr) / len(psnr))

    tq.close()

