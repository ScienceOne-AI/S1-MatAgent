import os
import glob
from tqdm import tqdm
import logging
import subprocess
from ase.io import write, read
from toolkits.hea import Boundary_X, Boundary_Y, absorb_h2o, absorb_h, absorb_oh_h

import socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
logger = logging.getLogger(__name__)

try:
    client.connect(('127.0.0.1', 40500))
    logger.info('Client start')
except Exception as e:
    logger.info('Server in mace_server.py not started')
    logger.error(e)
    exit(0)

def mace_predict(poscar_path, option) -> dict:
    logger.info(f'Calculating energy of {poscar_path}')
    if option == 'gradient':
        send_info = poscar_path+'####gradient'
        client.send(send_info.encode('utf-8'))
        response = client.recv(1024)
        energy, gradients = eval(response.decode('utf-8'))
        return energy, gradients
    else:
        if option == 'energy_ori':
            send_info = poscar_path+'####energy_ori'
        else:
            send_info = poscar_path+'####energy'
        client.send(send_info.encode('utf-8'))
        response = client.recv(1024)
        energy = float(response.decode('utf-8'))
        return energy, {}

def mace_predict_by_prefix(poscar_root_path, option, prefix=None, quite=True):
    """find all files with center prefix in their names, predict energy for them"""
    search_pattern = os.path.join(poscar_root_path, f"{prefix}*")
    # use glob to match files
    matched_files = glob.glob(search_pattern)
    # filter out directories
    poscar_paths = [f for f in matched_files if os.path.isfile(f)]

    site_info_dict = {}
    if quite:
        loop = poscar_paths
    else:
        loop = tqdm(poscar_paths, desc=f'mace predict with prefix {prefix}')
    for poscar_path in loop:
        #logger.info(f'Calculating energy of {poscar_path}')
        try:
            poscar_pure_path = os.path.basename(poscar_path)
            site_indice = poscar_pure_path.split(prefix)[1].split('_')
            site = int(site_indice[0])  # extract the number after the prefix in the filename

            if option == 'gradient':
                send_info = poscar_path+'####gradient'
                client.send(send_info.encode('utf-8'))
                response = client.recv(1024)
                energy, gradients = eval(response.decode('utf-8'))
                if len(site_indice) > 1:
                    index = site_indice[1]
                    if site not in site_info_dict.keys() or energy>site_info_dict[site][0]:
                        site_info_dict[site]=(energy, gradients)
                else:
                    site_info_dict[site]=(energy, gradients)
            else:
                if option == 'energy_ori':
                    send_info = poscar_path+'####energy_ori'
                else:
                    send_info = poscar_path+'####energy'
                client.send(send_info.encode('utf-8'))
                response = client.recv(1024)
                energy = float(response.decode('utf-8'))            
                if len(site_indice) > 1:
                    index = site_indice[1]
                    if site not in site_info_dict.keys() or energy>site_info_dict[site][0]:
                        site_info_dict[site]=(energy, {})
                else:
                    site_info_dict[site]=(energy, {})
        except subprocess.CalledProcessError as e:
            d = {"Command failed with return code": e.returncode, "Error output": e.stderr}
            logger.error(d)
            return []
    site_info_list = [(site,info[0],info[1]) for site,info in site_info_dict.items()]
    return site_info_list

class AlkalineHER:
    def __init__(self, slab_poscar_path, poscar_root_path, min_possible_sites_number=Boundary_X*Boundary_Y//4):
        os.makedirs(poscar_root_path, exist_ok=True)
        self.slab_poscar = slab_poscar_path
        self.slab = read(slab_poscar_path)
        self.poscar_root_path = poscar_root_path
        self.slab_energy = None
        self.slab_gradient = None
        self.get_slab_info()
        self.possible_sites = []
        self.site_energys = {}  # slab*H2O各位点能量
        self.site_gradients = {}  # 吸附水各位点焓变对元素的梯度
        self.min_possible_sites_number = min_possible_sites_number
        self.h_site_h2o = {}


    def get_slab_info(self):
        option = 'gradient'
        e, g = mace_predict(self.slab_poscar, option = option)
        #print("slab_energy: ", e)
        #print("slab_gradient: ", d)
        self.slab_energy = e
        self.slab_gradient = g
    

    def cal_absorb_H2O_deltaH(self, return_gradient_sum=False):
        '''if return_gradient_sum=False, gradient_sum={e:0 for e in self.slab_gradient.key()}, won't change self.site_gradients'''
        self.h_site_h2o = absorb_h2o(self.slab, self.poscar_root_path)
        assert len(self.possible_sites) == 0
        assert len(self.site_energys) == 0
        assert len(self.site_gradients) == 0
        if not (self.slab_gradient and self.slab_energy):
            self.get_slab_info()
        if return_gradient_sum:
            option = 'gradient'
        else:
            option = 'energy'
        site_energy_gradient_list = mace_predict_by_prefix(self.poscar_root_path, option=option, prefix='POSCAR_absorb_H2O_')
        delta_h_sum = 0
        gradient_sum = {e:0 for e in self.slab_gradient.keys()}
        for site,energy,gradient in site_energy_gradient_list:
            if gradient:
                self.site_gradients[site] = gradient
                e_d = dict(gradient)
                for e,d in e_d.items():
                    gradient_sum[e] = gradient_sum[e] + d - self.slab_gradient[e]
            
            self.site_energys[site] = energy
            
            delta_h = energy - self.slab_energy + 14.22
            delta_h_sum += delta_h
            #print(f"step1 delta H for site {site}: {delta_h}")
            if delta_h < 0:  # 0
                self.possible_sites.append(site)
        if len(self.possible_sites) < self.min_possible_sites_number:
            status = False
            message = f'The ability to adsorb H2O is weak, and only {len(self.possible_sites)} of the {Boundary_X*Boundary_Y} sites on the surface have the ability to adsorb H2O up to the standard. '
        else:
            status = True
            message = 'The ability to adsorb H2O meets the standard. '
        return {'status': status, 'message': message, 'delta_h_sum': delta_h_sum, 'gradient_sum': gradient_sum}

    def cal_H2O_decompose_deltaH(self, compute_all=True, return_gradient_sum=False):
        '''compute_all=True则计算所有位点，否则只计算step1筛选出的self.possible_sites'''
        if compute_all==False and len(self.possible_sites) == 0:
            return False
        count = 0
        if compute_all:
            possible_sites = [p for p in self.site_energys.keys()]
        else:
            possible_sites = self.possible_sites
        absorb_oh_h(self.slab, possible_sites, self.poscar_root_path, self.h_site_h2o)
        if return_gradient_sum:
            option = 'gradient'
        else:
            option = 'energy'

        assert self.slab_gradient
        site_energy_gradient_list = mace_predict_by_prefix(self.poscar_root_path, option = option, prefix='POSCAR_absorb_OH_H_')
        gradient_sum = {e:0 for e in self.slab_gradient.keys()}
        delta_h_sum = 0
        count = 0
        for site, energy, gradient in site_energy_gradient_list:
            if gradient:
                e_d = dict(gradient)
                for e,d in e_d.items():
                    gradient_sum[e] = gradient_sum[e] + d - self.site_gradients[site][e]

            delta_h = energy - self.site_energys[site]
            delta_h_sum += delta_h
            #print(f"step2 delta H for site {site}: {delta_h}")
            if delta_h < 0:  # 0
                count += 1
        if count < self.min_possible_sites_number:
            status = False
            message = f'The ability to promote the decomposition of H2O into H and OH is weak. Only {count} of the {len(possible_sites)} sites that can adsorb H2O can promote the decomposition of H2O into H and OH. '
        else:
            status = True
            message = 'The ability to promote the decomposition of H2O into H and OH meets the standard. '
        
        return {'status': status, 'message': message, 'delta_h_sum': delta_h_sum, 'gradient_sum': gradient_sum}

    def cal_absorb_H(self, return_gradient_sum=False, square_delta_h=True):
        absorb_h(self.slab, self.poscar_root_path)
        if return_gradient_sum:
            option = 'gradient'
        else:
            option = 'energy'
        site_energy_gradient_list = mace_predict_by_prefix(self.poscar_root_path, option = option, prefix='POSCAR_absorb_H_')
        gradient_sum = {e:0 for e in self.slab_gradient.keys()}
        count1 = 0
        count2 = 0
        delta_h_sum = 0
        for site, energy, gradient in site_energy_gradient_list:
            delta_h = energy - self.slab_energy + 3.4 + 0.06
            if gradient:
                self.site_gradients[site] = gradient
                e_d = dict(gradient)
                          
                for e,d in e_d.items():
                    if square_delta_h:
                        gradient_sum[e] = gradient_sum[e] + 2*(d - self.slab_gradient[e])
                    else:
                        if delta_h>=0:
                            gradient_sum[e] = gradient_sum[e] + (d - self.slab_gradient[e])
                        else:
                            gradient_sum[e] = gradient_sum[e] - (d - self.slab_gradient[e])

            
            if square_delta_h:
                delta_h_sum += delta_h*delta_h
            else:
                delta_h_sum += abs(delta_h)
            #print(f"step3 delta H for site {site}: {delta_h}")
            if delta_h > 0.5:
                count1+=1
            elif delta_h < 0.5:
                count2+=1
        if count1+count2<=Boundary_X*Boundary_Y/2:
            status = True
            message = 'It is easy to adsorb H and the adsorbed H is easy to be released, which meets the requirements'
        elif count1 >= count2:
            status = False
            message = 'The ability to adsorb H is too weak and needs to be strengthened'
        else:
            status = False
            message = 'The ability to adsorb H is too strong and needs to be weakened'

        return {'status': status, 'message': message, 'delta_h_sum': delta_h_sum, 'gradient_sum':gradient_sum}
    

    def evaluate(self, step2_compute_all=True, step3_square_delta_h=True, return_gradient_sum=False):
        d1 = dict(self.cal_absorb_H2O_deltaH(return_gradient_sum))
        #print(d1)
        d2 = dict(self.cal_H2O_decompose_deltaH(step2_compute_all, return_gradient_sum))
        #print(d2)
        d3 = dict(self.cal_absorb_H(return_gradient_sum, step3_square_delta_h))
        #print(d3)
        return d1, d2, d3
