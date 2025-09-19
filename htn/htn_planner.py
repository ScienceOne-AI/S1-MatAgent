import asyncio
from htn.htn_prompt import *
from typing import List, Optional, Any
from camel.tasks.task import Task, TaskManager, TaskState
from camel.societies.workforce.utils import TaskResult
from camel.models import BaseModelBackend
from camel.agents import ChatAgent
from camel.logger import get_logger, set_log_level
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits import HumanToolkit
from camel.toolkits.function_tool import FunctionTool
import os
import json
from pprint import pformat
import logging
logger = get_logger(__name__)
set_log_level(logging.INFO)

async def yield_text(text, speed = 0.001):
    i = 0
    for word in text.split(" "):
        await asyncio.sleep(speed)
        yield word + " "
    '''
    while i < len(text):
        yield text[i]
        i += 1
        await cl.sleep(speed)
    '''

def update_plan(task):
    with open('todo.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if(task in lines[i]):
                if('- [ ]' in lines[i]):
                    lines[i] = lines[i].replace('- [ ]', '- [X]')
                else:
                    lines[i] = lines[i][:(4 - int(i == 0))] + ' ✅' + lines[i][(4 - int(i == 0)):]

    updated_plan = ''.join(lines)
    with open('todo.md', 'w', encoding='utf-8') as f:
        f.write(updated_plan)

    return updated_plan

def parse_llm_json_response(response_text: str) -> dict:
    """Parse JSON response from LLM, handling various formats and cleanup"""
    logger.debug(f"Parsing raw response: {response_text}")

    # Remove markdown code blocks
    if "```" in response_text:
        parts = response_text.split("```")
        for part in parts:
            if "{" in part and "}" in part:
                response_text = part
                break

    # Remove any leading/trailing non-JSON content
    start_idx = response_text.find("{")
    end_idx = response_text.rfind("}") + 1
    if start_idx != -1 and end_idx != 0:
        response_text = response_text[start_idx:end_idx]

    # Clean up common formatting issues
    response_text = response_text.replace("\n", " ")
    response_text = response_text.replace("    ", " ")
    response_text = " ".join(response_text.split())

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"Initial JSON parse failed: {e}")
        # Try additional cleanup
        response_text = response_text.replace("\\", "")
        response_text = response_text.replace('""', '"')
        return json.loads(response_text)

class HTNTask(Task):
    name: Optional[str] = None
    task_type: Optional[str] = None
    agent: Optional[dict] = None
    dependency: Optional[str] = None
    predecessors: list = [] #  dependent task are the predecessors
    cl_task: Any = None

    def __init__(self,
            content: str,
            id: str = "",
            name: Optional[str] = None,
            task_type: Optional[str] = None,
            agent: Optional[dict] = None,
            dependency: Optional[str] = None,
            predecessors: list = [] #  dependent task are the predecessors
        ):
        super().__init__(id=id, content=content)
        self.name = name
        self.task_type = task_type
        self.agent = agent
        self.dependency = dependency
        self.predecessors = predecessors
        if(os.environ.get('START_CHAINLIT', 0)):
            import chainlit as cl
            self.cl_task=cl.Task(title=self.name, status=cl.TaskStatus.READY)

    async def is_task_primitive(self, planning_agent: Optional[ChatAgent] = None) -> bool:
        r"""Determine is the given task is primitive task
        Args:
            planning_agent (ChatAgent): The planning agent.
        Returns:
            is_primitive (Boolean): Whether the task is primitive.
        """
        if(self.task_type is None):
            msg = HTN_TASK_PRIMITIVE_PROMPT.format(
                content=self.content,
                predecessor_tasks_info=self.get_tasks_info('predecessor'),
                capacity=self.agent['system_message']
            )
            response = await planning_agent.astep(msg)
            response = response.msg.content.lower()
            self.task_type = 'primitive' if 'primitive' in response else 'compound'
        return self.task_type == 'primitive'

    async def adecompose(
        self,
        planning_agent: Optional[ChatAgent] = None,
        tool_description: Optional[str] = 'No Avaliable Tools',
    ) -> List["HTNTask"]:
        r"""Decompose a task to a list of sub-tasks. It can be used for data
        generation and planner of agent.

        Args:
            planning_agent (ChatAgent): The planning agent.
        Returns:
            List[Task]: A list of tasks which are :obj:`Task` instances.
        """
        msg = HTN_TASK_DECOMPOSITION_WITH_CAPACITIES_PROMPT.format(
            content=self.content,
            predecessor_tasks_info=self.get_tasks_info('predecessor'),
            tool_description=tool_description,
        )
        response = await planning_agent.astep(msg)
        format_response = parse_llm_json_response(response.msg.content)
        subtasks = []
        dependency = format_response['dependency']
        predecessors = self.predecessors
        for i, (task_name, task_content) in enumerate(list(format_response.items())[1:]):
            if (i >= 1 and dependency == 'sequence'):
                predecessors.append(subtasks[-1])
            subtask = HTNTask(id=f"{self.id}.{i}", predecessors=predecessors.copy(), name= task_name, content=task_content['content'], agent={'system_message': task_content['system_message'], 'tools': task_content['tools']})
            subtasks.append(subtask)
        #self.predecessors += subtasks
        self.dependency = dependency
        for task in subtasks:
            task.additional_info = self.additional_info
        return subtasks

    def get_tasks_info(self, task_type: str):
        assert task_type in ['predecessor', 'subtask']
        if(task_type == 'predecessor'):
            tasks = self.predecessors
            task_tag = 'predecessor_task'
        else:
            tasks = self.subtasks
            task_tag = 'subtask'

        if (len(tasks)):
            tasks_info = [
                f"<{task_tag}>\n    <id>{task.id}</id>\n <content>{task.content}</content>\n  <result>{task.result}</result>\n</{task_tag}>" if task.result else
                f"<{task_tag}>\n    <id>{task.id}</id>\n <content>{task.content}</content>\n</{task_tag}>"
                for task in tasks
            ]
            tasks_info = "\n".join(tasks_info)
        else:
            tasks_info = f'<{task_tag}>None</{task_tag}>'
        return tasks_info

class HTN:
    r"""Hierarchical Task Network is used to manage tasks.

        Args:
            task (HTNTask): The root Task.
            model (BaseModelBackend): The planning model.
            max_depth (int): The maximum depth of the hierarchy.
        """
    def __init__(self, task: str, planning_model: BaseModelBackend, tools: List[FunctionTool], answerer_model: BaseModelBackend=None, max_depth: int=2, root_task_prompt: str=HTN_ROOT_TASK_PROCESS_PROMPT):
        self.root_task_prompt: str = root_task_prompt
        self.max_depth: int = max_depth
        self.planning_agent: ChatAgent = ChatAgent(HTN_TASK_PLANNER_SYSMSG, model=planning_model, tools=[*HumanToolkit().get_tools()])
        self.root_task: HTNTask = HTNTask(name='root_task', content=task, agent={'system_message': BASIC_LLM_CAPACITIES, 'tools': [tool.openai_tool_schema['function']['name'] for tool in tools]})
        self.task_manager: TaskManager = TaskManager(self.root_task)
        if(int(os.environ.get('START_CHAINLIT', 0))):
            import chainlit as cl
            self.cl_task_list = cl.TaskList()
            self.cl_task_list.status = "Running..."
        if(answerer_model):
            self.answerer_agent = ChatAgent(
                system_message="You are a helpful assistant that can answer questions and provide final answers.",
                model=answerer_model
            )
        else:
            self.answerer_agent = None
        self.tools: List[FunctionTool] = tools


    async def get_final_answer(self):
        self.answerer_agent.reset()

        subtask_info = ""
        for subtask in self.root_task.subtasks:
            subtask_info += f"Subtask {subtask.id}: {subtask.content}\n"
            subtask_info += f"Subtask {subtask.id} result: {subtask.result}\n\n"

        prompt = f"""
        I am solving a question and have obtained the primary answer.
        <question>
        {self.root_task.content}
        </question>
        <primary_answer>
        {self.root_task.result.content}
        </primary_answer>

        Now, I have solved the question by decomposing it into several subtasks, the subtask information is as follows:
        <subtask_info>
        {subtask_info}
        </subtask_info>

        Now, I need you to determine the final answer. Do not try to solve the question, just pay attention to ONLY the format in which the answer is presented. DO NOT CHANGE THE MEANING OF THE PRIMARY ANSWER.
        You should first analyze the answer format required by the question and then output the final answer that meets the format requirements. 
        Here are the requirements for the final answer:
        <requirements>
        The final answer must be output exactly in the format specified by the question. The final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
        If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. Numbers do not need to be written as words, but as digits.
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. In most times, the final string is as concise as possible (e.g. citation number -> citations)
        If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        </requirements>

        Please output with the final answer according to the requirements without any other text. If the primary answer is already a final answer with the correct format, just output the primary answer.
        """

        resp = self.answerer_agent.step(prompt)
        return resp.msg.content


    async def build_htn(self):
        if(int(os.environ.get('START_CHAINLIT', 0))):
            import chainlit as cl
            async with cl.Step(name="Building HTN..", type="htn", default_open=True, show_input=False) as step:
                # generate agents accordingly binding to task

                await step.stream_token(
                    f"🚀 Building **Hierarchical Task Network (HTN)** for task planning.",
                )
                logger.info(f"⏳ Start Decomposing {self.root_task.id}: {self.root_task.content}")
                markdown_plan = await self.decompose_task(self.root_task, 0, self.max_depth, '')
                logger.info(f"✅ Finished Task Decomposition")
                logger.info(f"📝 Plan Created\n\n{markdown_plan}")
                with open('todo.md', 'w', encoding='utf-8') as f:
                    f.write(markdown_plan)

                name = '📝 Plan Created: todo.md'
                element = cl.Text(name=name, content=markdown_plan, display="side")
                await cl.Message(
                    content = name,
                    elements = [element]
                ).send()

                step.output = markdown_plan
        else:
            logger.info(f"⏳ Start Decomposing {self.root_task.id}: {self.root_task.content}")
            markdown_plan = await self.decompose_task(self.root_task, 0, self.max_depth, '')
            logger.info(f"✅ Finished Task Decomposition")
            logger.info(f"📝 Plan Created\n\n{markdown_plan}")
            with open('todo.md', 'w', encoding='utf-8') as f:
                f.write(markdown_plan)

    async def decompose_task(self, task:HTNTask, cur_depth, max_depth, markdown_plan:str):
        if(int(os.environ.get('START_CHAINLIT', 0))):
            import chainlit as cl
            async with cl.Step(name=f"Start decomposing {task.name}", type="decomposition", default_open=True, show_input=False) as step:
                self.planning_agent.reset()
                if (cur_depth == 0):
                    markdown_plan += f"{'#' * (cur_depth + 3)} **{task.content}**"
                    markdown_plan += '\n\n' if (cur_depth == 0) else '\n'
                else:
                    markdown_plan += f"{'  ' * (cur_depth - 1)}- [ ] **{task.name}**: {task.content}\n"
                if(cur_depth >= max_depth or await task.is_task_primitive(self.planning_agent)):
                    task.task_type = 'primitive'
                    return markdown_plan
                else:
                    self.planning_agent.reset()

                    tool_description = ""
                    for tool in self.tools:
                        tool_description += f"{tool.openai_tool_schema['function']['name']}: {tool.openai_tool_schema['function']['description']}\n"

                    subtasks = await task.adecompose(planning_agent=self.planning_agent, tool_description=tool_description)
                    await self.cl_task_list.send()
                    self.task_manager.add_tasks(subtasks)
                    for subtask in subtasks:
                        logger.info(f"🧾 Subtask {subtask.id}: {subtask.content}")
                        task.add_subtask(subtask)
                        await self.cl_task_list.add_task(subtask.cl_task)

                    for subtask in subtasks:
                        markdown_plan = await self.decompose_task(subtask, cur_depth+1, max_depth, markdown_plan)
                    #task.markdown_plan = markdown_plan
                return markdown_plan
        else:
            self.planning_agent.reset()
            if (cur_depth == 0):
                markdown_plan += f"{'#' * (cur_depth + 3)} **{task.content}**"
                markdown_plan += '\n\n' if (cur_depth == 0) else '\n'
            else:
                markdown_plan += f"{'  ' * (cur_depth - 1)}- [ ] **{task.name}**: {task.content}\n"
            if(cur_depth >= max_depth or await task.is_task_primitive(self.planning_agent)):
                task.task_type = 'primitive'
                return markdown_plan
            else:
                self.planning_agent.reset()

                tool_description = ""
                for tool in self.tools:
                    tool_description += f"{tool.openai_tool_schema['function']['name']}: {tool.openai_tool_schema['function']['description']}\n"

                subtasks = await task.adecompose(planning_agent=self.planning_agent, tool_description=tool_description)
                self.task_manager.add_tasks(subtasks)
                for subtask in subtasks:
                    logger.info(f"🧾 Subtask {subtask.id}: {subtask.content}")
                    task.add_subtask(subtask)

                for subtask in subtasks:
                    markdown_plan = await self.decompose_task(subtask, cur_depth+1, max_depth, markdown_plan)
            return markdown_plan

    async def htn_planning(self, task:HTNTask) -> str:
        # start planning on task
        #logger.info(f'start planning {task.id}: {task.name}')
        task.set_state(TaskState.OPEN)

        if(task.task_type == 'compound'):
            if(task.dependency == 'sequence'):
                for subtask in task.subtasks:
                    await self.htn_planning(subtask)
            else:
                async with asyncio.TaskGroup() as tg:
                    for subtask in task.subtasks:
                        tg.create_task(self.htn_planning(subtask))

        # aggregating predecessors
        while(True):
            # wait for predecessors DONE
            is_predecessors_done = True
            for predecessor in task.predecessors:
                if(predecessor.state in ['OPEN', 'RUNNING']):
                    is_predecessors_done = False
            if(is_predecessors_done):
                break
        if(int(os.environ.get('START_CHAINLIT', 0))):
            result = await self.execute_task_cl(task)
        else:
            result = await self.execute_task(task)
        if(not task.parent and self.answerer_agent):
            result.content = self.get_final_answer()

        task.update_result(result.content)
        if(result.failed):
            pass
        else:
            return result.content

    async def execute_task(self, task:HTNTask) -> TaskResult:
        logger.info(f'🚀 Start executing{task.id}: {task.name}')
        task.set_state(TaskState.RUNNING)
        predecessor_tasks_info = task.get_tasks_info('predecessor')

        if(task.parent):
            if(task.task_type == 'primitive'):
                prompt = HTN_PRIMITIVE_TASK_PROCESS_PROMPT.format(
                    content=task.content,
                    predecessor_tasks_info=predecessor_tasks_info,
                    root_task=self.root_task.content,
                )
            elif(task.task_type == 'compound'):
                prompt = HTN_COMPOUND_TASK_PROCESS_PROMPT.format(
                    content=task.content,
                    predecessor_tasks_info=predecessor_tasks_info,
                    subtasks_info=task.get_tasks_info('subtask'),
                    root_task=self.root_task.content,
                )
        else:
            # root task prompt
            prompt = self.root_task_prompt.format(
                content=task.content,
                predecessor_tasks_info=predecessor_tasks_info,
                subtasks_info=task.get_tasks_info('subtask'),
                root_task=self.root_task.content,
            )

        try:
            # build task agent
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4_1,
                model_config_dict={"temperature": 0},
            )
            agent_tools = []
            for selected_tool in task.agent['tools']:
                for available_tool in self.tools:
                    if(selected_tool == available_tool.openai_tool_schema['function']['name']):
                        agent_tools.append(available_tool)

            task_agent = ChatAgent(task.agent['system_message'], model=model, tools=agent_tools)
            agent_info = {"🧾 task": task.content, '💬 sys_msg': task.agent['system_message'], '⚒️ tools': task.agent['tools']}
            logger.info(f'Building Agent: \n{pformat(agent_info)}')

            response = await task_agent.astep(prompt, response_format=TaskResult)
        except Exception as e:
            logger.error(
                f"Error occurred while processing task {task.id}:"
                f"\n{e}"
            )
            return TaskResult({"content": f"Error occurred while processing task: {e}", "failed": True})

        result_dict = json.loads(response.msg.content)
        task_result = TaskResult(**result_dict)

        task.result = task_result.content
        logger.info(f'✅Finished executing {task.id}: {task.name}')
        logger.info(f'📄Result: {task.result}')
        updated_plan = update_plan(task.content)
        logger.info(f"📝Plan Updated\n\n{updated_plan}")

        if(task_result.failed):
            task.set_state(TaskState.FAILED)
        else:
            task.set_state(TaskState.DONE)
        return task_result

    async def execute_task_cl(self, task:HTNTask) -> TaskResult:
        if(int(os.environ.get('START_CHAINLIT', 0))):
            import chainlit as cl
        async with cl.Step(name=f"Executing {task.name}", type="execution", default_open=True, show_input=False) as step:
            step.input = task.name
            logger.info(f'🚀 Start executing{task.id}: {task.name}')
            task.set_state(TaskState.RUNNING)
            task.cl_task.status = cl.TaskStatus.RUNNING
            await self.cl_task_list.send()
            predecessor_tasks_info = task.get_tasks_info('predecessor')

            if(task.parent):
                if(task.task_type == 'primitive'):
                    prompt = HTN_PRIMITIVE_TASK_PROCESS_PROMPT.format(
                        content=task.content,
                        predecessor_tasks_info=predecessor_tasks_info,
                        root_task=self.root_task.content,
                    )
                elif(task.task_type == 'compound'):
                    prompt = HTN_COMPOUND_TASK_PROCESS_PROMPT.format(
                        content=task.content,
                        predecessor_tasks_info=predecessor_tasks_info,
                        subtasks_info=task.get_tasks_info('subtask'),
                        root_task=self.root_task.content,
                    )
            else:
                # root task prompt
                prompt = self.root_task_prompt.format(
                    content=task.content,
                    predecessor_tasks_info=predecessor_tasks_info,
                    subtasks_info=task.get_tasks_info('subtask'),
                    root_task=self.root_task.content,
                )
            try:
                # build task agent
                # agent={'system_message': task_content['system_message'], 'tools': task_content['tools']}
                model = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI,
                    model_type=ModelType.GPT_4_1,
                    model_config_dict={"temperature": 0},
                )
                agent_tools = []
                for selected_tool in task.agent['tools']:
                    for available_tool in self.tools:
                        if(selected_tool == available_tool.openai_tool_schema['function']['name']):
                            agent_tools.append(available_tool)

                async with cl.Step(name=f"Building Agent", type="agent", default_open=True, show_input=False, language="json") as agent_step:
                    task_agent = ChatAgent(task.agent['system_message'], model=model, tools=agent_tools)
                    agent_info = {"🧾 task": task.content, '💬 sys_msg': task.agent['system_message'], '⚒️ tools': task.agent['tools']}
                    agent_step.output = json.dumps(agent_info, ensure_ascii=False, indent=4)
                    #print(json.dumps(agent_info, indent=4))
                    logger.info(f'Building Agent: \n{pformat(agent_info)}')

                response = await task_agent.astep(prompt, response_format=TaskResult)
            except Exception as e:
                logger.error(
                    f"Error occurred while processing task {task.id}:"
                    f"\n{e}"
                )
                return TaskResult({"content": f"Error occurred while processing task: {e}", "failed": True})

            result_dict = json.loads(response.msg.content)
            task_result = TaskResult(**result_dict)

            task.result = task_result.content
            logger.info(f'✅Finished executing {task.id}: {task.name}')
            logger.info(f'📄Result: {task.result}')
            updated_plan = update_plan(task.content)
            logger.info(f"📝Plan Updated\n\n{updated_plan}")

            name = '📝 Plan Updated: todo.md'
            element = cl.Text(name=name, content=updated_plan, display="side")
            await cl.Message(
                content=name,
                elements=[element]
            ).send()

            if(task.parent):
                #step.stream_token(task_result)

                async for text in yield_text(task_result.content, speed=1e-2):
                    await step.stream_token(text)

            else:
                step.output = ''

            #step.output = task_result.content

            if(task_result.failed):
                task.set_state(TaskState.FAILED)
                task.cl_task.status = cl.TaskStatus.FAILED
            else:
                task.set_state(TaskState.DONE)
                task.cl_task.status = cl.TaskStatus.DONE
            await self.cl_task_list.send()

        return task_result

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()
    from toolkits import DomainToolkit
    from camel.toolkits import CodeExecutionToolkit, BrowserToolkit

    print(os.environ.get('START_CHAINLIT', 0))

    models = {
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_5,
            model_config_dict={"temperature": 0},
        ),
    }
    tools = [
        *BrowserToolkit(
            headless=False,
        ).get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *DomainToolkit(cache_dir='tmp/file/').get_tools(),
    ]

    htn = HTN(
        task="Compile a report on GPT 4.1",
        planning_model=models['planning'],
        tools=tools)
    asyncio.run(htn.build_htn())
    asyncio.run(htn.htn_planning(htn.root_task))
