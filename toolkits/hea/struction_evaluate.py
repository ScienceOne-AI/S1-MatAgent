from toolkits.hea import prepare_slab, random_change_atoms, AlkalineHER
import os
from dotenv import load_dotenv
load_dotenv()
import logging
logger = logging.getLogger(__name__)

elements = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]
metallic_elements = [
    "Li", "Be",
    "Na", "Mg", "Al", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Og"
]

def standard_element_ratios2str(element_ratios):
    result = ''
    for p in element_ratios:
        result += p[0] + str(p[1]*4)
    return result


def evaluate_alloy_workflow(element_ratios):
    try:
        max_element_ratio = 0
        max_element = None
        for p in element_ratios:
            if p[0] not in elements:
                return f'{p[0]} is not a correct symbol for any element!'
            if p[0] not in metallic_elements:
                return f'{p[0]} is not a metallic element!'
            if not isinstance(p[1], int):
                return f'Expect positive int but got {p[1]} (type is {type(p[1])})'
            if p[1] <= 0:
                return f'Expect positive int but got {p[1]}'
            if p[1] > max_element_ratio:
                max_element_ratio = p[1]
                max_element = p[0]

        namestr = standard_element_ratios2str(element_ratios)
        found = False
        for entry in os.listdir('tmp/'):
            if 'evaluate_' + namestr == entry:
                found = True
                break
        current_path = os.getcwd()
        if not found:
            slab = prepare_slab(symbol=max_element)
            slab = random_change_atoms(slab, element_ratios, poscar_path='tmp/evaluate_slab.symbols/POSCAR_slab')  # slab.symbols is a special symbol that will be replaced; here we already makedirs
            
            slab_poscar_path = os.path.join(current_path, f'tmp/evaluate_{slab.symbols}/POSCAR_slab')
            poscar_root_path = os.path.join(current_path, f'tmp/evaluate_{slab.symbols}/')
        else:
            slab_poscar_path = os.path.join(current_path, f'tmp/evaluate_{namestr}/POSCAR_slab')
            poscar_root_path = os.path.join(current_path, f'tmp/evaluate_{namestr}/')
        
        mechanism = AlkalineHER(slab_poscar_path, poscar_root_path)
    except KeyError as e:
        logger.error(str(e))
        return 'KeyError: '+str(e)+"Pay attention to the format! For example, [('Fe', 1), ('Ni', 2), ('Pt', 6)] means Fe:Ni:Pt=1:2:6"
    except Exception as e:
        logger.error(str(e))
        return str(e)
    return mechanism.evaluate(step2_compute_all=True, step3_square_delta_h=False, return_gradient_sum=True)
