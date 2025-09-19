import numpy as np
import random
import asyncio
import socket
import logging
from asyncio import AbstractEventLoop
from ase import units
from ase.md.langevin import Langevin
from mace.calculators import MACECalculator
from ase.calculators.calculator import Calculator
from ase.io import write
from ase.data import chemical_symbols, atomic_numbers
import ase
import torch


try:
    calc = MACECalculator(model_paths="model/mace_finetuned.model", default_dtype="float64", device='cuda:0')
    calc_ori = MACECalculator(model_paths="model/mace_agnesi_medium.model", default_dtype="float64", device='cuda:1')
except Exception as e:
    calc = MACECalculator(model_paths='large')
    calc_ori = MACECalculator(model_paths='large')


def compute_element_gradients(atoms_path, candidate=['Ni', 'Co', 'Ru', 'Pt', 'Fe', 'Mo', 'Cu', 'Pd', 'Ir', 'Rh']):
    # load atoms
    atoms = ase.io.read(atoms_path)
    Calculator.calculate(calc, atoms)
    batch_base = calc._atoms_to_batch(atoms)
    batch = calc._clone_batch(batch_base)
    model = calc.models[0]
    
    # build computational graph
    batch = calc._clone_batch(batch_base)
    batch["node_attrs"] = batch["node_attrs"].detach().clone().requires_grad_()
    
    with torch.set_grad_enabled(True):
        out = model(
            batch.to_dict(),
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
        )

        grad = torch.autograd.grad(out["energy"], batch["node_attrs"])[0]
    
    my_atomic_numbers = atoms.get_atomic_numbers()
    z_table = calc.z_table
    
    # calculate average gradient for each element type
    element_gradients = {}
   
    for e in candidate:
        i = atomic_numbers[e]
        element_indices = [j for j, an in enumerate(my_atomic_numbers) if an == i]  # 哪些原子是该元素
        if element_indices:
            element_grad = grad[element_indices, z_table.zs.index(i)].mean().item()
            element_gradients[e] = element_grad
        
    return out["energy"].item(), element_gradients


async def echo(connection: socket.socket, loop: AbstractEventLoop):
    try:
        while data := await loop.sock_recv(connection, 1024):
            if not data: 
                raise Exception('network error')
            data = data.decode('utf-8')
            print(f"Received data {data}")
            
            datas = data.split("####")
            assert len(datas) == 2
            print(datas[0])

            if datas[1]=='gradient':
                response = compute_element_gradients(datas[0])
            elif datas[1]=='energy_ori':
                atoms = ase.io.read(datas[0], format='vasp')
                atoms.calc = calc_ori
                response = atoms.get_potential_energy()
            else:
                atoms = ase.io.read(datas[0], format='vasp')
                atoms.calc = calc
                response = atoms.get_potential_energy()
            
            print(f"Returned data {response}")
            await loop.sock_sendall(connection, str(response).encode('utf-8'))
    except Exception as e:
        logging.exception(e)

    finally:
        connection.close()


async def listen_for_connection(server_socket=None, loop: AbstractEventLoop=None):

    while True:
        connection, address = await loop.sock_accept(server_socket)
        connection.setblocking(False)
        print(f" Get a connection from {address}")
        asyncio.create_task(echo(connection, loop))

async def main(ip, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = (ip, port)
    server_socket.setblocking(False)
    server_socket.bind(server_address)
    server_socket.listen(5)


    await listen_for_connection(server_socket, loop=asyncio.get_event_loop())


asyncio.run(main('127.0.0.1',40500))

