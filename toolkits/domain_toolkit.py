from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool

from typing import List, Optional, Tuple, Literal
from urllib.parse import urlparse
import subprocess

import re
import os
from camel.logger import get_logger, set_log_level
import logging
logger = get_logger(__name__)
set_log_level(logging.INFO)

import asyncio

from toolkits import mace_predict
from dotenv import load_dotenv
load_dotenv()

class DomainToolkit(BaseToolkit):
    r"""A class representing a toolkit for the design and evaluation of material structures.

    """

    def __init__(
        self, cache_dir: str
    ) -> None:
        f"""Initializes the DomainToolkit. Files will be processed in {cache_dir}."""
        super().__init__()
        logger.info(
            f"DomainToolkit initialized. Files will be processed in {cache_dir}."
        )
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.count = 0

    async def get_poscar_from_materials_project(self, formula: str) -> str:
        r"""Get the POSCAR structure file of material according to its chemical formula.

            Args:
                formula (str): Chemical formula of the material.

            Returns:
                str: The absolute path of the POSCAR structure file generated.
        """
        from mp_api.client import MPRester
        from toolkits import remove_quotes

        formula = remove_quotes(formula)
        output_path = self.cache_dir
        with MPRester(os.getenv("MP_API_KEY")) as mpr:
            structure = mpr.get_structures(chemsys_formula=formula)
            if (structure):  # 获取与化学式相同的Structure对象列表
                structure = structure[0]
                dir = os.path.join(output_path, formula)
                os.makedirs(dir, exist_ok=True)
                path = os.path.join(dir, 'POSCAR')
                structure.to(fmt='poscar', filename=path)
                logger.info(f"Saving the POSCAR file to {path}")
            else:
                return {f'Error: {formula} is not valid in material project'}
        return path

    async def mace_predict_energy(self, poscar_path) -> str:
        r"""Calculates the energy of the material based on its structure file located at poscar_path.

        Args:
            poscar_path (str): The path to the structure file

        Returns:
            float: A result string demonstrating the result of the calculation.

        """
        if(poscar_path == 'POSCAR'):
            poscar_path = 'tmp/file/Ni(OH)2/POSCAR'
        logger.info(f'Calculating energy of {poscar_path}')
        result = mace_predict(poscar_path, 'energy_ori')
        return result

    def _str2elementratios(self, component: str, total_ratio: int=16):
        # use regex to match element name and number
        pattern = r'([A-Z][a-z]*)(\d*)'
        matches = re.findall(pattern, component)
        
        elements = []
        for element, num_str in matches:
            # if num_str is empty, count is 1
            count = int(num_str) if num_str else 1
            if(element == 'Ni'):
                elements = [(element, count)] + elements
            else:
                elements.append((element, count))

        # Proportionally mapping to total_ratio atoms
        # Calculate the total sum of the current ratios
        cur_total_ratio = sum(ratio for _, ratio in elements)

        # Calculate the scaling factor
        scaling_factor = total_ratio / cur_total_ratio

        # Scale the ratios to be proportional and round to nearest integer
        scaled_elements = [(element, round(ratio * scaling_factor)) for element, ratio in elements]

        # Ensure the sum equals exactly the target sum (16), adjust the `Ni` element if necessary
        new_sum = sum(ratio for _, ratio in scaled_elements)
        difference = total_ratio - new_sum

        if difference != 0:
            scaled_elements[0] = (scaled_elements[0][0], scaled_elements[0][1] + difference)

        return scaled_elements
    
    def _elementratios2str(self, component: List[Tuple[str,int]], total_ratio: int=64):
        result = ""
        sum_ratio = sum([num for element, num in component])

        for element, num in component:
            result+=element+str(num*total_ratio//sum_ratio)
        
        return result

    async def optimize_alloy_activity(self, composition: str) -> str:
        r"""Enhance the alloy activity by performing local search based on the initial proportion of alloy materials to find optimal alloy compositions with higher activity.
            It is userful to enhance the activity of high entropy alloy (HEA) catalyst based on its initial composition.

        Args:
            composition (str): The initial alloy component. For example: NiPtCoMoRh.

        Returns:
           str: A result string demonstrating the iteration process of alloy component.

        """
        from toolkits import change_step, order_by_periodic_table
        try:
            initial_element_ratios = self._str2elementratios(composition)
            results_info = ""
            cnt = 0
            cur_element_ratios = initial_element_ratios

            while True:
                cnt += 1
                new_element_ratios, cur_activity, new_activity = change_step(cur_element_ratios)  # self.my_change_step(cur_element_ratios)
                if(cnt == 1):
                    initial_activity = cur_activity
                    results_info += f'Starting to optimize **{composition}** Initial Activity: **{initial_activity}**'
                    if(int(os.environ.get('START_CHAINLIT', 0))):
                        import chainlit as cl
                        await  cl.context.current_step.stream_token(results_info)   

                if cur_activity == new_activity:
                    results_info += f"\n**Search Finished** \n Final Composition: **{cur_element_ratios}** \n Catalyst Activity: **{cur_activity}**\nActivity promotion=**{new_activity-initial_activity}**"
                    if(int(os.environ.get('START_CHAINLIT', 0))):
                        await cl.context.current_step.stream_token(results_info)
                    break
                else:
                    results_info += f"\n**Iteration {cnt}**: \n**{cur_element_ratios}: {cur_activity}**\t↩️\n**{new_element_ratios}:{new_activity}**\nActivity promotion=**{new_activity-cur_activity}**\n"
                    logger.info(results_info)
                    cur_element_ratios = new_element_ratios
                    if(int(os.environ.get('START_CHAINLIT', 0))):
                        await cl.context.current_step.stream_token(results_info)
                        cl.sleep(1)

            new_composition = self._elementratios2str(new_element_ratios, total_ratio=16)

            path = f'tmp/evaluate_{self._elementratios2str(new_element_ratios)}/POSCAR_slab'
            results_info += f'\nThe structure file of the new composition **{new_composition}** is saved in `{path}`.'
            
            return results_info

        except Exception as e:
            logger.error(f"Error: {e}")
            return f"Error: {e}"

    async def demonstrate_alloy_structure(self, path: str):
        r"""Visualize the alloy structure given its structure file path

            Args:
                path (str): The path to the structure file

            Returns:
                result(str): demonstration result
          """
        import os
        env = os.environ.copy()
        env['DISPLAY'] = ':1'
        if(int(os.environ.get('START_CHAINLIT', 0))):
            import chainlit as cl
            mycomponent = cl.CustomElement(
                name="Virtue Computer",
                display='side'
            )
            await cl.Message(content="Virtue Computer", elements=[mycomponent]).send()

            structure_name = f"Structure File"

            await cl.Message(
                content='Structure file saved:',
                elements=[cl.File(name=structure_name, path=path, display="inline")]
            ).send()

        await asyncio.create_subprocess_exec(
            os.getenv('VESTA_PATH'), path, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)


    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the functions in the toolkit.
        """
        return [
            FunctionTool(self.get_poscar_from_materials_project),
            FunctionTool(self.mace_predict_energy),
            FunctionTool(self.optimize_alloy_activity),
            FunctionTool(self.demonstrate_alloy_structure),
        ]