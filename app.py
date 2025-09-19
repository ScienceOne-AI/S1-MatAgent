import chainlit as cl
from htn.htn_planner import HTN
from run import build_htn_args
from toolkits import yield_word
from dotenv import load_dotenv

load_dotenv()

args = build_htn_args()

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Material Q&A",
            message="Summarize the latest application of alloy materials in the catalytic field",
            icon="/public/icon/search.png",
        ),

        cl.Starter(
            label="Material Calculation",
            message="Calculate the energy of Ni(HO)2, and demonstrate its structure",
            icon="/public/icon/calculation.png",
        ),
        cl.Starter(
            label="Material Design",
            message="Design highly active high entropy alloy (HEA) catalysts for hydrogen evolution reaction (HER) in an alkaline environment based on the literature file",
            icon="/public/icon/catalyst.png",
        ),
    ]

def extract_files_from_text(text:str):
    import re
    import os
    pattern = r"`\s*(.*?)\s*`"
    matches = re.findall(pattern, text)
    elements = []
    files = []
    for match in list(set(matches)):
        if(os.path.exists(match)):
            file = cl.File(name=match,path=match,display="inline")
            files.append(match)
            elements.append(file)

    if(elements):
        msg = cl.Message(content="File List\n", elements=elements)
        return msg
    else:
        return None

@cl.on_message
async def main(message: cl.Message):

    import os
    os.environ['START_CHAINLIT'] = '1'
    content = message.content.strip()

    if(len(message.elements) and isinstance(message.elements[0], cl.element.File)):
        content = content + f'\n I have uploaded a file: `{message.elements[0].path}`' 
    elif message.content == "Design highly active high entropy alloy (HEA) catalysts for hydrogen evolution reaction (HER) in an alkaline environment based on the literature file":
        files = None
        while files is None:
            files = await cl.AskFileMessage(
                content="upload a literature file (.csv) to continue",
                accept=[".csv"],
                max_size_mb=10
            ).send()

        # get the first file user uploaded
        content = content + f'\n I have uploaded a file: `{files[0].path}`'

    htn = HTN(
        task=content,
        planning_model=args['models']['planning_model'],
        tools=args['tools'],
    )
    await htn.build_htn()
    response = await htn.htn_planning(htn.root_task)
    htn.cl_task_list.status = "Done"
    await htn.cl_task_list.send()
    del htn

    msg = cl.Message(content="")
    async for text in yield_word(response, 0.05):
        await msg.stream_token(text)

    msg = extract_files_from_text(response)
    if(msg):
        await msg.send()
    print(response)
    return response
