from typing import Dict, List, Optional, Any
from camel.logger import get_logger, set_log_level
from camel.toolkits.base import BaseToolkit
from camel.toolkits import CodeExecutionToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.models import  BaseModelBackend
from camel.agents import ChatAgent
from pydantic import BaseModel, Field

import pandas as pd
import re
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import os
from camel.utils import LiteLLMTokenCounter
import chainlit as cl
from datetime import datetime

logger = get_logger(__name__)
set_log_level(logging.INFO)
from dotenv import load_dotenv
load_dotenv()

class FilterResult(BaseModel):
    r"""The result of a task."""

    sorted_elements: str = Field(description="The top 10 elements and its frequencies in descending order")
    literature_file: str = Field(description="The path of the extracted literatures.")
    image_file: str = Field(description="The path of the plotted images.")

  
def order_by_periodic_table(component):
    element_order = {'Ni':0, 'Co':27, 'Ru':44, 'Pt':78, 'Fe':26, 'Mo':42, 'Cu':29, 'Pd':46, 'Ir':77, 'Rh':45}  # 'Ni':28
    component_sorted = sorted(component, key=lambda x: element_order.get(x[0], 255))
    return component_sorted

class LiteratureToolkit(BaseToolkit):
    """
        A toolkit for retrieve information from literatures
    """

    def __init__(self) -> None:
        r"""Initializes the LiteratureToolkit."""
        super().__init__()
        logger.info(
            "LiteratureToolkit initialized"
        )
        recommender_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            url='https://uni-api.cstcloud.cn/v1',
            api_key=os.getenv("ScienceOne_API_KEY"),
            model_type='S1-Base-Pro',
            model_config_dict={"temperature": 0},
        )

        coder_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4_1,
            model_config_dict={"temperature": 0},
        )

        code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
        self.recommender_agent = ChatAgent(
            system_message="You are a materials expert skilled in providing advice in hydrogen evolution reactions(HER) related fields.",
            model=recommender_model,
            tools=code_runner_toolkit.get_tools()
        )
        self.coder_agent = ChatAgent(
            system_message="You are a materials expert skilled in coding.",
            model=coder_model,
            tools=code_runner_toolkit.get_tools()
        )

    async def extract_alloy_compositions_from_literatures(self, literature_path:str):
        r"""For literature data processing.
            This function extracts alloy compositions from the literatures, and finally returns the extracted literature file path.

            Args:
                literature_path (str): The path to the literature file.

            Returns:
                Dict[str]: Filtered literature path.

        """
        # Load literature file
        df = pd.read_csv(literature_path)
        df['date'] = pd.to_datetime(df['date'])

        metal_dict = {
            'lithium': 'Li', 'sodium': 'Na', 'potassium': 'K', 'rubidium': 'Rb', 'cesium': 'Cs',
            'caesium': 'Cs', 'francium': 'Fr',

            'beryllium': 'Be', 'magnesium': 'Mg', 'calcium': 'Ca', 'strontium': 'Sr', 'barium': 'Ba', 'radium': 'Ra',

            'scandium': 'Sc', 'yttrium': 'Y', 'titanium': 'Ti', 'zirconium': 'Zr', 'hafnium': 'Hf', 'vanadium': 'V',
            'niobium': 'Nb', 'tantalum': 'Ta', 'chromium': 'Cr', 'molybdenum': 'Mo', 'tungsten': 'W', 'manganese': 'Mn',
            'technetium': 'Tc', 'rhenium': 'Re', 'iron': 'Fe', 'ruthenium': 'Ru', 'osmium': 'Os', 'cobalt': 'Co',
            'rhodium': 'Rh', 'iridium': 'Ir', 'nickel': 'Ni', 'palladium': 'Pd', 'platinum': 'Pt', 'copper': 'Cu',
            'silver': 'Ag', 'gold': 'Au', 'zinc': 'Zn', 'cadmium': 'Cd', 'mercury': 'Hg',

            'lanthanum': 'La', 'cerium': 'Ce', 'praseodymium': 'Pr', 'neodymium': 'Nd', 'promethium': 'Pm', 'samarium': 'Sm',
            'europium': 'Eu', 'gadolinium': 'Gd', 'terbium': 'Tb', 'dysprosium': 'Dy', 'holmium': 'Ho', 'erbium': 'Er',
            'thulium': 'Tm', 'ytterbium': 'Yb', 'lutetium': 'Lu',

            'actinium': 'Ac', 'thorium': 'Th', 'protactinium': 'Pa', 'uranium': 'U', 'neptunium': 'Np', 'plutonium': 'Pu',
            'americium': 'Am', 'curium': 'Cm', 'berkelium': 'Bk', 'californium': 'Cf', 'einsteinium': 'Es', 'fermium': 'Fm',
            'mendelevium': 'Md', 'nobelium': 'No', 'lawrencium': 'Lr',

            'aluminum': 'Al', 'aluminium': 'Al',
            'gallium': 'Ga', 'indium': 'In', 'thallium': 'Tl', 'tin': 'Sn', 'lead': 'Pb', 'bismuth': 'Bi', 'polonium': 'Po',
        }

        def _replace_metals(text):
            """
            Use regex to replace metals in the text, only match full words to avoid misreplacement.
            """
            def _match_func(match):
                word = match.group(0)
                replacement = metal_dict.get(word.lower())
                return replacement if replacement is not None else word

            # re.IGNORECASE, "Iron" and "iron"
            pattern = r'\b(' + '|'.join(re.escape(name) for name in metal_dict.keys()) + r')\b'
            result_text = re.sub(pattern, _match_func, text, flags=re.IGNORECASE)
            return result_text
        
        df['abstract'] = df['abstract'].apply(_replace_metals)

        ELEMENTS = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Sn', 'Sb', 'Te', 'Xe',
            'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
            'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        ]
        # delete 'K','Na','I','In','V','As' because of ambiguity
        
        # build regex pattern to match element symbols (sorted by length in descending order to match longer symbols first)
        elements_sorted = sorted(ELEMENTS, key=len, reverse=True)

        def _is_valid_formula(formula_str):
            """
            Check if the string is a valid chemical formula.

            Args:
                formula_str (str): The string to be checked.

            Returns:
                bool: True if the string is a valid chemical formula, False otherwise.
            """
            # remove hyphens (handle compound formulas)
            formula_str = formula_str.replace('-', '')

            # empty string is not a valid formula
            if not formula_str:
                return False

            i = 0
            n = len(formula_str)

            while i < n:
                matched = False
                for elem in elements_sorted:
                    if formula_str.startswith(elem, i):
                        # found
                        i += len(elem)
                        matched = True

                        # check if there is a number after the element symbol
                        if i < n and formula_str[i].isdigit():
                            # skip the number part (including decimal point)
                            while i < n and (formula_str[i].isdigit() or formula_str[i] == '.'):
                                i += 1
                        break

                if not matched:
                    return False

            # if successfully processed the entire string, it is a valid formula
            return i == n


        def _extract_chemical_formulas(text):
            """
            Extract chemical formulas from the text.

            Args:
                text (str): The input text.

            Returns:
                list: A list of extracted chemical formulas.
            """
            # extract all possible chemical formulas (including uppercase letters, numbers, dots, and hyphens)
            potential_formulas = re.findall(r'\b[A-Z][A-Za-z0-9.-]+\b', text)

            # validate each extracted string to check if it is a valid chemical formula
            valid_formulas = []
            for formula in potential_formulas:
                if _is_valid_formula(formula):
                    # process compound formulas (remove hyphens)
                    processed_formula = formula.replace('-', '')
                    valid_formulas.append(processed_formula)

            return list(set(valid_formulas))


        # Extract alloy compositions from abstract
        alloy_literatures = []
        for idx, row in df.iterrows():
            abstract = str(row['abstract'])
            compositions = _extract_chemical_formulas(abstract)
            #print(compositions)
            if compositions:
                entry = row.to_dict()
                entry['alloy_composition'] = compositions
                alloy_literatures.append(entry)
            else:
                #print(idx)
                entry = row.to_dict()
                entry['alloy_composition'] = []
                alloy_literatures.append(entry)

        # Save alloy literatures to CSV
        alloy_df = pd.DataFrame(alloy_literatures)
        alloy_csv_path = 'alloy_compositions.csv'
        alloy_df.to_csv(alloy_csv_path, index=False)


        logger.info(f"📊 {alloy_csv_path}")
        literature_name = f"📊 Literature Saved: {alloy_csv_path}"

        df = pd.read_csv(alloy_csv_path)
        if(int(os.environ.get('START_CHAINLIT', 0))):
            import chainlit as cl
            elements = [
                cl.Dataframe(name=literature_name, data=df[:10], display="side"),
                cl.File(name=alloy_csv_path, path=alloy_csv_path, display="inline"),
            ]
            await cl.Message(
                content=f"{literature_name}",
                elements=elements
            ).send()
        return {
            "extraction_file_path": alloy_csv_path,
        }
    
    async def count_frequency_of_metal_elements(self):
        r"""For literature data processing. This function is used to analyse elements' frequency in column 'alloy_composition' of 'alloy_compositions.csv' file.
            Count the frequency of metal elements in alloy compositions extracted. This helps to narrow down the scope of high-entropy alloy composition recommendation and optimization.

            Returns:
                Dict[str]: Top ten metal elements and the bar chart path.

        """
        extraction_file_path = 'alloy_compositions.csv'
        image_file = 'top10_elements.png'
        prompt = f"""Write a complete Python code and execute it to complete the following tasks:
                1.List all metal elements directly.
                2.Read {extraction_file_path} to DataFrame.
                3.For each article, consider each chemical formula in column 'alloy_composition' and count the frequency of occurrence of each metallic element as part of a chemical formula. Each metal element counted at most once per article.
                4.Draw a bar chart for the top 10 metal elements with the highest frequency and save it as '{image_file}'."""
        
        temp_results = await self.coder_agent.astep(prompt)        
        logger.info(temp_results.msg.content)

        self.coder_agent.reset()

        logger.info(f"🖼️ {image_file}")
        image_name = f"🖼️ Image Saved: {image_file}"
        if(int(os.environ.get('START_CHAINLIT', 0))):
            import chainlit as cl
            elements = [
                cl.Image(name=image_name, path=image_file, display="side"),
                cl.File(name=image_file,path=image_file,display="inline"),
            ]
            await cl.Message(
                content=f"{image_name}",
                elements=elements
            ).send()
        return temp_results.msg.content
    
    async def recommend_alloy_compositions_from_literatures(self):
        r"""Recommend high-entropy alloy compositions using statistically identified high-frequency metal elements based on literature information. Given a query with specific material properties,
            this function will recommend initial composition based on information from the literatures, and return the alloy compositions.
            It is useful for recommend alloy compositions with specific properties in hydrogen evolution reactions(HER) related field.

            Returns:
                str: Alloy compositions recommended.

        """
        literature_path = 'alloy_compositions.csv'
        df = pd.read_csv(literature_path).dropna()
        df['alloy_composition'] = df['alloy_composition'].apply(lambda x: eval(x))
        df['authors'] = df['authors'].apply(lambda x: str(x))
        df['index'] = df.index
        def _generate_reference(row):
            authors = row['authors']
            # process author format
            if ';' in authors:
                author_list = authors.split('; ')
                if len(author_list) > 1:
                    formatted_authors = ', '.join(author_list[:-1]) + ', and ' + author_list[-1]
                else:
                    formatted_authors = authors
            else:
                formatted_authors = authors

            year = row['date'].split('-')[0] if pd.notna(row['date']) else 'n.d.'
            title = row['title'] if pd.notna(row['title']) else 'No title'
            publication = row['publications'] if pd.notna(row['publications']) else 'No publication'
            reference = f"{formatted_authors} ({year}). {title}. {publication}."
            return reference

        df['reference'] = df.apply(_generate_reference, axis=1)
        
        temp_contents = ''
        token_counter = LiteLLMTokenCounter(model_type=ModelType.QWEN_TURBO)
        system_prompt_token = len(token_counter.encode(self.recommender_agent.system_message.content))
        recommend_num_per_part = 3
        base_prompt_front = """
            Currently, it is necessary to construct a nickel-based high-entropy alloy catalyst for the hydrogen evolution reaction under alkaline conditions. 
            You need to select another 4 elements from these 9 options to pair with Ni, forming a 5-component high-entropy alloy formula without any other elements. The 9 options are: Pt, Co, Fe, Mo, Ru, Cu, Ir, Pd, Rh. 
            All recommended formulas should be derived from the integration and synthesis of information and insights across multiple literature sources provided below:
                """
        base_prompt_back = f"""
            ----------------------------------
            Return exact {recommend_num_per_part} different results strictly in the following format:
            ---
            1. **[chemical formula of composition]**
            - **Rationale**: [recommended reason, don't cite the reference indices]
            - **References**: [all the indices of reference literatures in listing]
            ---
        """
        base_prompt_token = len(token_counter.encode(base_prompt_front+base_prompt_back))
        all_literature_information = [{'index': row['index'], 'abstract': row['abstract']} for _, row in df.iterrows()]
        all_literature_token = len(token_counter.encode(json.dumps(all_literature_information)))
        limit = 32000  # ScienceOne
        num_parts = all_literature_token // (limit-base_prompt_token-system_prompt_token) + 1
        logger.info(f"Number of parts: {num_parts}")

        for i in range(num_parts):
            logger.info(f"Part {i}")
            if i < num_parts-1:
                literature_information = all_literature_information[i*len(df)//num_parts:(i+1)*len(df)//num_parts]
            else:
                literature_information = all_literature_information[(num_parts-1)*len(df)//num_parts:]
            prompt = base_prompt_front + str(json.dumps(literature_information)) + base_prompt_back

            temp_results = await self.recommender_agent.astep(prompt)        
            temp_contents += temp_results.msg.content
            logger.info(temp_results.msg.content)
            logger.info('----------')

            self.recommender_agent.reset()

        check_code="""
        ```
            import pandas as pd
            import re

            def is_valid_chemical_formula(formula):
                valid_elements = {'Ni', 'Co', 'Ru', 'Pt', 'Fe', 'Mo', 'Cu', 'Pd', 'Ir', 'Rh'}
                pattern = re.compile(r'([A-Z][a-z]*)(\d*\.?\d+)?')
                matches = pattern.findall(formula)
                elements = []
                subscripts = []
                for match in matches:
                    element = match[0]
                    subscript = match[1] if match[1] else '1'
                    elements.append(element)
                    try:
                        subscripts.append(float(subscript))
                    except ValueError:
                        return False
                if not all(e in valid_elements for e in elements):
                    return False
                if len(elements) != len(set(elements)):
                    return False
                if 'Ni' not in elements:
                    return False
                if len(elements) != 5:
                    return False
                reconstructed = ''.join(f"{e}{s if s != 1.0 else ''}" for e, s in zip(elements, subscripts))
                if reconstructed != formula:
                    return False
                return True

            def rewrite_formula(formula):
                formula = formula.replace("-", "")
                pattern = re.compile(r'([A-Z][a-z]*)(\d*\.?\d+)?')
                matches = pattern.findall(formula)
                elements = []
                subscripts = []
                for match in matches:
                    element = match[0]
                    subscript = match[1] if match[1] else '1'
                    elements.append(element)
                    subscripts.append(float(subscript))
                
                sort_order = {'Ni':0, 'Co':27, 'Ru':44, 'Pt':78, 'Fe':26, 'Mo':42, 'Cu':29, 'Pd':46, 'Ir':77, 'Rh':45}
                paired = list(zip(elements, subscripts))
                paired.sort(key=lambda x: sort_order[x[0]])
                
                new_formula_parts = []
                for element, subscript in paired:
                    if subscript == 1.0:
                        new_formula_parts.append(element)
                    else:
                        new_formula_parts.append(f"{element}{subscript}")
                return ''.join(new_formula_parts)

            # ToDo, for example data=[['NiPtCoRuRh', '**Pt/Co** synergies improve HER activity; **Ru/Rh** reduce hydrogen adsorption energy', [205, 211, 248, 258]]
            data = [fill in all recommended alloy compositions here]
            df = pd.DataFrame(data, columns=['formula', 'rationale', 'references'])
            df['formula'] = df['formula'].apply(rewrite_formula)
            valid_df = df[df['formula'].apply(is_valid_chemical_formula)].copy()
            
            valid_df.to_csv('recommend_compositions.csv', index=False)
        ```
        """


        recommend_file = 'recommend_compositions.csv'
        prompt = f"""
        There are some recommended alloy compositions:
        {temp_contents}
        ----------------------------------
        There are some codes for reference:
        {check_code}
        ----------------------------------
        You need to write and execute a Python code implementation containing the following three steps:
        1. Convert the recommended {recommend_num_per_part*num_parts} alloy compositions to DataFrame with keys `formula`, `rationale` and `references`. Listing {recommend_num_per_part*num_parts} alloy compositions completely and accurately rather than just providing examples.
        2. Filter out invalid alloy compositions using the validation code;
        3. Rewrite the chemical formula by sorting elements according to ascending values in sort_order;
        4. Merge same compositions. Merge the content of the `formula` field and combine the individual `references` lists into one (e.g. combine [1, 5, 68] and [109, 133] to get [1, 5, 68, 109, 133]);
        5. Save final results to `{recommend_file}` with keys `formula`, `rationale` and `references`.
        """
        results = await self.coder_agent.astep(prompt)
        
        df2 = pd.read_csv(recommend_file, encoding='utf-8')
        newline = '\n'
        cltext = ''
        for i, row in df2.iterrows():
            cltext += f"""{i:>2}. **{row['formula']}**
        - **Rationale**: {row['rationale']}
        - **References**: {f"{newline}     - ".join([f"{[j]} {df['reference'][index]}" for j,index in enumerate(eval(row['references']))])}

            """       
        if(int(os.environ.get('START_CHAINLIT', 0))):
            import chainlit as cl
            elements = [
                cl.Text(content=cltext, display="inline"),
                cl.File(name=recommend_file, path=recommend_file, display="inline"),
            ]
            await cl.Message(
                content=cltext+f"\n{recommend_file}",
                elements=elements
            ).send()
        
        results = ",".join([c for c in df2['formula']])
        return results


    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the
        functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects
                representing the functions in the toolkit.
        """
        return [
            FunctionTool(self.extract_alloy_compositions_from_literatures),
            FunctionTool(self.count_frequency_of_metal_elements),
            FunctionTool(self.recommend_alloy_compositions_from_literatures),
        ]


if __name__ == "__main__":
    rag_toolkit = LiteratureToolkit()

    import asyncio
    #l = asyncio.run(rag_toolkit.extract_alloy_compositions_from_literatures("literatures.csv"))
    #l = asyncio.run(rag_toolkit.count_frequency_of_metal_elements())
    l = asyncio.run(rag_toolkit.recommend_alloy_compositions_from_literatures())
    print(l)
