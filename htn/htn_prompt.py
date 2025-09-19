from camel.prompts import TextPrompt

HTN_TASK_PLANNER_SYSMSG = "You are a hierarchical task planning expert."

HTN_TASK_CLARIFY_PROMPT = TextPrompt(
"""You have been given to process the following task:
    {content}
==============================
Please determine whether the description is clear, unambiguous, and contains sufficient detail.
Guidelines:
 - If the task description is clear, output the original task as is;
 - If the task description contains ambiguity, vagueness, or lacks completeness, ask the user for clarification, and output the clarified task.
 - You MUST answer only the final task (either ambiguous or not).
"""
)

HTN_TASK_PRIMITIVE_PROMPT = TextPrompt(
"""Given current task, its predecessor tasks and your capacity:
<current_task>
    {content}
</current_task>
{predecessor_tasks_info}
<capacity>
    {capacity}
</capacity>
==============================
Assume all predecessor tasks are done, determine if the current task is 'primitive' which can be achieved directly by your capacity and cannot be broken up into subtasks further, or 'compound' which can NOT be achieved directly and MUST be broken down into subtasks more.
Guidelines:
 - IF the current task relies on the results of predecessor tasks, you should assume all the predecessor tasks are DONE and the results are available now.
 - If the current task relies on processing data from information sources (such as paper, website), you should regard this task as 'compound'.
 - If the current task contains recommend and optimize, you should regard this task as 'compound'.
 - Provide the answer as 'primitive' or 'compound'.
"""
)

HTN_TASK_DECOMPOSITION_PROMPT = TextPrompt("""As a Task Decomposer, your objective is to divide the given task into subtasks.
You have been provided with the following objective:
{content}
Please format the subtasks as a numbered list within <tasks> tags, as demonstrated below:
<tasks>
<task>Subtask 1</task>
<task>Subtask 2</task>
</tasks>
Guidelines:
 - You MUST divide at least 2 subtasks for given task.
 - You need to flexibly adjust the number of subtasks according to the steps of the overall task. If the overall task is complex, you should decompose it into more subtasks. Otherwise, you should decompose it into less subtasks (e.g. 2-3 subtasks).
 - Each subtask should be significantly simpler than the original.
 - Each subtask should be concise, concrete, achievable and have clear completion criteria.
 - Ensure that the task plan is created without asking any questions.
 - Ensure sub-tasks collectively cover the entire original task.
 - Be specific and clear, do not add additional information (such as number of subtasks).
"""
)

HTN_TASK_DECOMPOSITION_WITH_CAPACITIES_PROMPT = TextPrompt(
"""As a Task Decomposer, your objective is to divide the current task into subtasks and assign it to an agent that is capable of solving it.
==============================
You have been provided with the current task and its predecessors tasks:
<current_task>
    {content}
</current_task>
{predecessor_tasks_info}
==============================
The agent is build upon a basic language model with additional tools to enhance its capabilities. 
The capability of the basic language model without tool is:
A basic language model is designed to assist with a wide range of tasks through natural language interaction, including content creation, summarization, translation, etc.
==============================
You have the following tools to use:
{tool_description}
==============================
According to the available tools and given task, please format your response in a JSON object, as demonstrated below:
{{
    "dependency": "the dependency of the subtasks, should be either sequence or parallel".
    "SUBTASK1": {{
        "content": "The content of the SUBTASK1", 
        "system_message": "The system message passed to language models to explain the role and capabilities of the agent.",
        "tools": ["tool1", "tool2"] # Tools containing the tool chosen from above list
   }}
}}
==============================
Guidelines:
 - You MUST divide AT LEAST 2 subtasks for given task.
 - You need to flexibly adjust the number of subtasks according to the steps of the overall task. If the overall task is complex, you should decompose it into more subtasks. Otherwise, you should decompose it into less subtasks (e.g. 2-3 subtasks).
 - You MUST determine the dependency of the generated subtasks in either 'sequence' (meaning the subtasks should be executed one by one) or 'parallel' (meaning the subtasks can be executed simultaneously) , if the dependency is 'sequence', provide the subtask in ordered sequence.
 - Each subtask should be significantly simpler than the original.
 - Each subtask should be concise, concrete, achievable by the agent and have clear completion criteria.
 - You MUST modify the key "SUBTASK1" in the JSON object to a short CAPITALIZED name reflecting its functionality.
 - Generate a system_message for each agent to explain its role and capacity. DO NOT explicitly mention what tools to use in the system_message, just provide relevant necessary tools and let the agent decide what to do.
 - If the task requires code execution, it should be informed in the system_message to first write code and then execute it.
 - If the task involves processing a file, keep the filename as its complete path when referencing it.
 - You can choose multiple tools when needed.
 - When a task involves knowledge-based content (such as formulas, constants, or factual information), you must choose relevant tools to retrieve up-to-date and authoritative sources for verification.
 - Ensure to chose tool from the given list when building agent, you can keep it as emtpy list [] when no tool is needed.
 - Ensure that the task plan is created without asking any questions.
 - Ensure subtasks collectively cover the entire original task.
 - Your response should be the required JSON object ONLY, do not add additional information.
"""
)

HTN_TASK_PROCESS_PROMPT = TextPrompt(
"""You need to process the current task.
Here are results of some predecessor tasks that you can refer to:
{predecessor_tasks_info}
You ultimate goal is to solve the root task:
<root_task>
    {root_task}
<root_task>
The content of the current task that you need to process is:
<current_task>
    {content}
</current_task>
You are asked to return the result of the current task.
=====================================================
Guidelines:
- If you need to write code, never generate code like "example code", your code should be completely runnable and able to fully solve the task. You MUST execute the code after generating the code.
- If you are going to process local files, you should explicitly mention all the processed file path (especially extracted files in zip files) in your answer to let other agents know where to find the file.
- If you are going to count, distinguish whether the countable items refer to a specific item or a category.
- You SHOULD use results of the predecessor tasks if they are related/userful to solve the current task.
- Never forget you ultimate goal is to solve the root task, the current task you are solving is the part of the root task.
- DO NOT OVER use tool, you should prefer not using tools unless need it.
"""
)

HTN_PRIMITIVE_TASK_PROCESS_PROMPT = TextPrompt(
"""You need to process the current primitive task.
Here are results of some predecessor tasks that you can refer to:
{predecessor_tasks_info}
You ultimate goal is to solve the root task:
<root_task>
    {root_task}
<root_task>
=====================================================
The content of the current primitive task that you need to process is:
<current_task>
    {content}
</current_task>
You are asked to return the result of the current task.
=====================================================
Guidelines:
- If you need to write code, never generate code like "example code", your code should be completely runnable and able to fully solve the task. You MUST execute the code after generating the code.
- If you are going to process local files, you should explicitly mention all the processed file path (especially extracted files in zip files) in your answer to let other agents know where to find the file.
- If you are going to count, distinguish whether the countable items refer to a specific item or a category.
- When a task involves knowledge-based content (such as formulas, constants, or factual information), you must choose relevant tools to retrieve up-to-date and authoritative sources for verification.
- When the task brings up the information source (paper, website, etc.), you must visit the source to retrieve relevant information.
- You are suggested use results of the predecessor tasks if they are related/userful to solve the current task.
- Never forget you ultimate goal is to solve the root task, the current task you are solving is the part of the root task.
- DO NOT OVER use tool, you should prefer not using tools unless need it.
"""
)

HTN_COMPOUND_TASK_PROCESS_PROMPT = TextPrompt(
"""You need to process the current compound task.
Here are results of some predecessor tasks that you can refer to:
{predecessor_tasks_info}
Here are results of some subtasks that you can refer to:
{subtasks_info}
You ultimate goal is to solve the root task:
<root_task>
    {root_task}
<root_task>
=====================================================
The content of the current compound task that you need to process is:
<current_task>
    {content}
</current_task>
You are asked to return the result of the current task.
=====================================================
Guidelines:
- If you need to write code, never generate code like "example code", your code should be completely runnable and able to fully solve the task. You MUST execute the code after generating the code.
- If you are going to process local files, you should explicitly mention all the processed file path (especially extracted files in zip files) in your answer to let other agents know where to find the file.
- If you are going to count, distinguish whether the countable items refer to a specific item or a category.
- The compound task you are solving consists of the subtasks which has been solved before, you SHOULD aggregate the results of all subtasks if they are executed successfully instead of solving the compound task from ground.
- Never forget you ultimate goal is to solve the root task, the current task you are solving is the part of the root task.
- DO NOT OVER use tool, you should prefer not using tools unless need it.
"""
)

HTN_ROOT_TASK_PROCESS_PROMPT = TextPrompt(
"""You need to process the current root task.
Here are results of some predecessor tasks that you can refer to:
{predecessor_tasks_info}
Here are results of some subtasks that you can refer to:
{subtasks_info}
You ultimate goal is to solve the root task:
<root_task>
    {root_task}
<root_task>
=====================================================
The content of the current compound task that you need to process is:
<current_task>
    {content}
</current_task>
You are asked to return the result of the current task.
=====================================================
Guidelines:
- The root task you are solving consists of the subtasks which has been solved before, you SHOULD aggregate the results of all subtasks if they are executed successfully instead of solving the compound task from ground.
- Never forget you ultimate goal is to solve the root task, the current task you are solving is the part of the root task.
- Generate a summary in markdown when solving the root task to explain how you get the final result, add references to literatures or website for possible reference.
- DO NOT OVER use tool, you should prefer not using tools unless need it.
"""
)
HTN_TASK_VERIFIER_PROMPT = TextPrompt(
"""
You need to check carefully the process you solve the current task to make sure you have obtained the correct answer to the task.
You ultimate goal is to solve the root task:
<root_task>
    {root_task}
<root_task>
=====================================================
The content and result of the current task that you need to process is:
<current_task>
    {content}
</current_task>
<result>
    {task_result}
</result>
=====================================================
Here is the history action trajectory the task agent have taken to solve the current task:
{chat_history}
=====================================================
Now please carefully examine the results of the current task, and the history trajectories, and determine whether the task has been executed correctly(needs revision). If not, provide the reason and try solving the current task on you own.
=====================================================
Your output should be in json format, including the following fields:
- `if_need_revision`: bool, A boolean value indicating whether the task has been executed correctly (needs revision).
- `revision_reason`: str, The reason why the previous answer needs revision. If the task does not need revision, the value should be an empty string. 
- `revised_answer`: str, The revised answer for the task. If the task does not need revision, the value should be an empty string. 

Guidelines:
- If you need to write code, never generate code like "example code", your code should be completely runnable and able to fully solve the task. You MUST execute the code after generating the code.
- If you are going to process local files, you should explicitly mention all the processed file path (especially extracted files in zip files) in your answer to let other agents know where to find the file.
- If you are going to count, distinguish whether the countable items refer to a specific item or a category.
"""
)

BASIC_LLM_CAPACITIES = TextPrompt(
"""You are a basic language model with basic common sense designed to assist with daily tasks including content creation, summarization, translation, or write codes to solve complex tasks."""
)