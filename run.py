import asyncio
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from htn import HTN
import sys
from camel.toolkits import (
    SearchToolkit,
    FunctionTool,
    FileToolkit
)
from toolkits import DomainToolkit, LiteratureToolkit
from dotenv import load_dotenv
load_dotenv()

def build_htn_args():
    from dotenv import load_dotenv
    load_dotenv()
    
    planning_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1,
        model_config_dict={"temperature": 0},
    )

    search_toolkit = SearchToolkit()

    tools = [
        FunctionTool(search_toolkit.tavily_search),
        *FileToolkit().get_tools(),
        *DomainToolkit(cache_dir='tmp/file/').get_tools(),
        *LiteratureToolkit().get_tools(),
    ]

    args = {'models': {"planning_model": planning_model}, 'tools': tools}
    return args


async def main(task: str):
    args = build_htn_args()
    htn = HTN(
        task=task,
        planning_model=args['models']['planning_model'],
        tools=args['tools'],
    )
    await htn.build_htn()
    await htn.htn_planning(htn.root_task)

if __name__ == "__main__":
    import os
    os.environ['START_CHAINLIT'] = '0'

    args = sys.argv
    if(len(args) == 1):
        print('Provide the your task! e.g. python run.py your_task')
        exit(0)
    else:
        asyncio.run(main(args[1]))