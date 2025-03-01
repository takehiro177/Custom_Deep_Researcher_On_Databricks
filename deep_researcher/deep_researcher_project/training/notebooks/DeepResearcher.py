# Databricks notebook source
##################################################################################
# AI Agent Notebook
#
# This notebook shows an example of a development of AI Agent on Databricks.
#
# 
# 
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip install -U databricks-sdk
# MAGIC %pip install langgraph databricks-langchain
# MAGIC %pip install "mlflow-skinny[databricks]>=2.20.2" loguru

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %%writefile prompts.py

# Prompt to generate search queries to help with planning the report
report_planner_query_writer_instructions="""You are performing research for a report. 

<Report topic>
{topic}
</Report topic>

<Report organization>
{report_organization}
</Report organization>

<Task>
Your goal is to generate {number_of_queries} web search queries that will help gather information for planning the report sections. 

The queries should:

1. Be related to the Report topic
2. Help satisfy the requirements specified in the report organization

Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
</Task>
"""

# Prompt to generate the report plan
report_planner_instructions="""I want a plan for a report that is concise and focused.

<Report topic>
The topic of the report is:
{topic}
</Report topic>

<Report organization>
The report should follow this organization: 
{report_organization}
</Report organization>

<Context>
Here is context to use to plan the sections of the report: 
{context}
</Context>

<Task>
Generate a list of sections for the report. Your plan should be tight and focused with NO overlapping sections or unnecessary filler. 

For example, a good report structure might look like:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

Each section should have the fields:

- Name - Name for this section of the report.
- Description - Brief overview of the main topics covered in this section.
- Research - Whether to perform web research for this section of the report.
- Content - The content of the section, which you will leave blank for now.

Integration guidelines:
- Include examples and implementation details within main topic sections, not as separate sections
- Ensure each section has a distinct purpose with no content overlap
- Combine related concepts rather than separating them

Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow.
</Task>

<Feedback>
Here is feedback on the report structure from review (if any):
{feedback}
</Feedback>
"""

# Query writer instructions
query_writer_instructions="""You are an expert technical writer crafting targeted web search queries that will gather comprehensive information for writing a technical report section.

<Report topic>
{topic}
</Report topic>

<Section topic>
{section_topic}
</Section topic>

<Task>
Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information above the section topic. 

The queries should:

1. Be related to the topic 
2. Examine different aspects of the topic

Make the queries specific enough to find high-quality, relevant sources.
</Task>
"""

# Section writer instructions
section_writer_instructions = """You are an expert technical writer crafting one section of a technical report.

<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic>
{section_topic}
</Section topic>

<Existing section content (if populated)>
{section_content}
</Existing section content>

<Source material>
{context}
</Source material>

<Guidelines for writing>
1. If the existing section content is not populated, write a new section from scratch.
2. If the existing section content is populated, write a new section that synthesizes the existing section content with the Source material.
</Guidelines for writing>

<Length and style>
- Strict 150-200 word limit
- No marketing language
- Technical focus
- Write in simple, clear language
- Start with your most important insight in **bold**
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing 2-3 key items (using Markdown table syntax)
  * Or a short list (3-5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title : URL`
</Length and style>

<Quality checks>
- Exactly 150-200 words (excluding title and sources)
- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- One specific example / case study
- Starts with bold insight
- No preamble prior to creating the section content
- Sources cited at end
</Quality checks>
"""

# Instructions for section grading
section_grader_instructions = """Review a report section relative to the specified topic:

<Report topic>
{topic}
</Report topic>

<section topic>
{section_topic}
</section topic>

<section content>
{section}
</section content>

<task>
Evaluate whether the section content adequately addresses the section topic.

If the section content does not adequately address the section topic, generate {number_of_follow_up_queries} follow-up search queries to gather missing information.
</task>

<format>
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )
</format>
"""

final_section_writer_instructions="""You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic> 
{section_topic}
</Section topic>

<Available report content>
{context}
</Available report content>

<Task>
1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports: 
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point
</Task>

<Quality Checks>
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response
</Quality Checks>"""



# COMMAND ----------

# Define the Pydantic data class used in subsequent processing. It is primarily used to enforce structured output for the LLM.

from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(description="The content of the section.")

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Feedback(BaseModel):
    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )


# COMMAND ----------

# Define the utility functions used during agent processing. These functions mainly define searches using Tavily and format the search results.

# As for Tavily searches, they were implemented by reusing the Databricks connection object created in a previous article. Please refer to that article for more information on creating the connection object for Tavily.

# One change from the original repository is that the Tavily search is now a function that searches for a single query. In the original, it was a function that searched for multiple queries asynchronously. However, in this article, the implementation has been changed to perform parallel execution within LagnGraph.

import os
import asyncio
import requests
import mlflow
from loguru import logger

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ExternalFunctionRequestHttpMethod

def deduplicate_and_format_sources(
    search_responses, max_tokens_per_source, include_raw_content=True
):
    """ Formats a list of search responses into a readable string.
    Limits raw_content to approximately max_tokens_per_source.

    Args: 
        search_responses: A list of dictionaries containing each search response. Each dictionary includes: 
            - query: str 
            - results: A list of dictionaries. Each dictionary includes the following fields: 
                - title: str 
                - url: str 
                - content: str 
                - score: float 
                - raw_content: str|None 
        max_tokens_per_source: int 
        include_raw_content: bool

    Returns: 
        str: A formatted string containing sources with duplicates removed
    """
    # collect all sources from search responses
    sources_list = []
    for response in search_responses:
        if response:
            sources_list.extend(response["results"])

    # remove duplicates
    unique_sources = {source["url"]: source for source in sources_list}

    # format sources
    formatted_text = "Sources:\n\n"
    # use a char limit to prevent overly long responses
    char_limit = max_tokens_per_source * 2

    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        content = source["content"]
        # if content is too long, truncate it
        if len(content) > char_limit:
            content = content[:char_limit] + "... [truncated]"
        formatted_text += (
            f"Most relevant content from source: {content}\n===\n"
        )

        if include_raw_content:
            raw_content = source.get("raw_content")
            if raw_content is None:
                raw_content = ""
                logger.warning(
                    f"Warning: No raw_content found for source {source['url']}"
                )

            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """format a list of Section objects into a readable string"""
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
                            {'='*60}
                            Section {idx}: {section.name}
                            {'='*60}
                            Description:
                            {section.description}
                            Requires Research: 
                            {section.research}

                            Content:
                            {section.content if section.content else '[Not yet written]'}

                            """
    return formatted_str

@mlflow.trace
def tavily_search(query, include_raw_content=True, max_results=5):
    """Tavily API search function

    Args:
        query (str): Search query
        include_raw_content (bool): Whether to include raw content in the response
        max_results (int): Maximum number of search results to return

    Returns:
        dict: Search response dictionary with the following keys:
            - results (list): List of search results, each containing the following keys:
                - title (str):  Title of the search result
                - url (str):    URL of the search result
                - content (str):    Content of the search result
                - raw_content (str):    Raw content of the search result"""

    response = WorkspaceClient().serving_endpoints.http_request(
        conn="tavily_api",
        method=ExternalFunctionRequestHttpMethod.POST,
        path="search",
        json={
            "query": query,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "topic": "general",
        },
        headers={"Content-Type": "application/json"},
    )

    if not response.ok:
        return None

    return response.json()

def remove_thinking_text(text):
    """<think>  delete the text between <think> and </think> tags

    Args:
        text (str):     Text to process

    Returns:
        str:    resulting text with <think> and </think> tags removed
    """

    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]

    return text



# COMMAND ----------

from typing import Literal
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from databricks_langchain import ChatDatabricks

from loguru import logger

writer_model_name = "deepseek_r1_distilled_llama8b_v1"
planner_model_name = "deepseek_r1_distilled_llama8b_v1"

MAX_SEARCH_RESULT_FOR_SECTION_GENERATION = 2
MAX_SEARCH_RESULT_FOR_SECTION_COMPLETION = 4
MAX_TOKENS_PER_SOURCE = 800
MAX_WRITTING_TOKENS = 2000
MAX_PLANNING_TOKENS = 2000

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report based on the user-provided topic:

1. Introduction (No research needed)
    - Brief overview of the topic area

2. Content Sections:
    - Each section should focus on a subtopic of the user-provided topic

3. Conclusion
    - Aim for one structural element (list or table) summarizing the content sections
    - Provide a concise summary of the report"""

# COMMAND ----------
@task
def search_query(query: str, max_results: int = 2) -> dict:
    """Tavily API search function"""
    return tavily_search(query, max_results=max_results)

@task
def generate_queries(section: Section, number_of_queries: int = 2) -> list[SearchQuery]:
    """Generate search queries to help with planning the report """

    # generate search queries for the report planner
    writer_model = ChatDatabricks(
        model=writer_model_name,
        temperature=0,
    )
    structured_llm = writer_model.with_structured_output(Queries)

    system_instructions = query_writer_instructions.format(
        topic=section.name, section_topic=section.description, number_of_queries=number_of_queries
    )

    # generate search queries
    queries = structured_llm.invoke(
        [SystemMessage(content=system_instructions)]
        + [HumanMessage(content="Generate search queries to help with planning the report.")]
    )

    return queries.queries

@task
def write_section(section: Section, search_results: list) -> Section:
    """Write a section of the report based on the search results"""

    # if the sources in the search results are duplicated, remove duplicates and format the results into a single text
    source_str = deduplicate_and_format_sources(
        search_results,
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        include_raw_content=True,
    )

    # prepare the instructions for the section writer
    system_instructions = section_writer_instructions.format(
        section_title=section.name,
        section_topic=section.description,
        context=source_str,
        section_content=section.content,
    )

    # generate the section content
    writer_model = ChatDatabricks(
        model=writer_model_name,
        temperature=0,
        max_tokens=MAX_WRITTING_TOKENS,
    )
    section_content = writer_model.invoke(
        [SystemMessage(content=system_instructions)]
        + [HumanMessage(content="Write a section of the report based on the search results.")]
    )

    # update the section content
    section.content = section_content.content

    return section


@task
def grade_section(section: Section) -> Feedback:
    """Grade a section of the report"""

    # prepare the instructions for the section grader
    section_grader_instructions_formatted = section_grader_instructions.format(
        section_topic=section.description,
        section=section.content,
    )

    # grade the section
    writer_model = ChatDatabricks(model=writer_model_name, temperature=0)
    structured_llm = writer_model.with_structured_output(Feedback)
    feedback = structured_llm.invoke(
        [SystemMessage(content=section_grader_instructions_formatted)]
        + [HumanMessage(content="Grade a section of the report, and then generate follow-up search queries if needed.")]
    )

    return feedback


@entrypoint()
def build_section_with_web_research(inputs: dict):
    section = inputs["section"]
    max_loops_for_build_section = inputs.get("max_loops_for_build_section", 2)

    # initialize the section with a failing grade
    feedback = Feedback(
        grade="fail",
        follow_up_queries=generate_queries(section).result(),
    )

    # loop to build
    for _ in range(max_loops_for_build_section):
        if feedback.grade == "pass":
            break

        # after the web search, write the section and grade
        search_results = [
            search_query(
                query.search_query, max_results=MAX_SEARCH_RESULT_FOR_SECTION_COMPLETION
            )
            for query in feedback.follow_up_queries
        ]
        section = write_section(
            section,
            search_results=[r.result() for r in search_results],
        ).result()
        feedback = grade_section(section).result()

    return section


# COMMAND ----------

# test the build_section_with_web_research function

config = {"configurable": {"thread_id": "1"}}
section = Section(
    name="What is AI Agent?",
    description="Explain the concept of AI Agent and its applications.",
    research=True,
    content="",
)
result = build_section_with_web_research.invoke({"section": section}, config)

print(f"Name: {result.name}")
print(f"Description: {result.description}")
print(f"Content: {result.content}")


# COMMAND ----------


@task
def generate_report_plan(
    topic: str,
    feedback_on_report_plan: str = None,
    report_structure: str = DEFAULT_REPORT_STRUCTURE,
    number_of_queries: int = 2,
) -> List[Section]:
    """Generate search queries to help with planning the report"""

    # generate search queries for the report planner
    writer_model = ChatDatabricks(model=writer_model_name, temperature=0)
    structured_llm = writer_model.with_structured_output(Queries)

    # prepare the system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic,
        feedback=feedback_on_report_plan,
        report_organization=report_structure,
        number_of_queries=number_of_queries,
    )

    # generate search queries
    results = structured_llm.invoke(
        [SystemMessage(content=system_instructions_query)]
        + [HumanMessage(content="generate search queries to help with planning the report.")]
    )

    # web search queries
    query_list = [query.search_query for query in results.queries]
    return query_list

@task
def generate_sections(
    topic: str,
    search_results: List[dict],
    feedback_on_report_plan: str = None,
    report_structure: str = DEFAULT_REPORT_STRUCTURE,
) -> List[Section]:
    
    """
    Task (node) to generate sections of the report from search results

    Args:
        topic (str): Topic of the report
        search_results (List[dict]): List of search results
        feedback_on_report_plan (str, optional): Feedback on the report plan
        report_structure (str, optional): Structure of the report

    Returns:
        List[Section]: List of generated sections
    """

    # If the sources in the search results are duplicated, remove duplicates and format the results into a single text
    source_str = deduplicate_and_format_sources(
        search_results,
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        include_raw_content=False,
    )

    # Prepare the system prompt
    system_instructions_sections = report_planner_instructions.format(
        topic=topic,
        report_organization=report_structure,
        context=source_str,
        feedback=feedback_on_report_plan,
    )

    planner_llm = ChatDatabricks(model=planner_model_name, temperature=0.0, max_tokens=MAX_PLANNING_TOKENS)

    # Generate the sections
    structured_llm = planner_llm.with_structured_output(Sections)
    report_sections = structured_llm.invoke(
        [SystemMessage(content=system_instructions_sections)]
        + [
            HumanMessage(
                content="Please generate the sections of the report. \
                    The response should include a 'sections' field containing a list of sections. \
                    Each section should include name, description, plan, research, and content fields."
            )
        ]
    )

    sections = report_sections.sections

    return sections

@task
def human_feedback_for_sections(
    sections: Sections,
) -> tuple[bool, str]:
    """Get feedback on the report plan"""

    # Format the sections into a readable string
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan
    feedback = interrupt(
        f"Please provide feedback on the report plan below.\n\n{sections_str}\n\nDoes the report plan meet your needs? Pass 'true' to approve the report plan or provide feedback to regenerate the report plan."
    )

    # If the feedback is 'True', proceed with the subsequent processing. If some string is given, add its content to the report structure and replan
    if isinstance(feedback, bool):
        return True, None
    elif isinstance(feedback, str):
        return False, feedback
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")

@task
def build_completed_section(section: Section):
    """Build a completed section of the report"""

    return build_section_with_web_research.invoke(
        {
            "section": section,
            "max_loops_for_build_section": 2,
        }
    )

@task
def gather_completed_sections(completed_sections) -> str:
    """Task (node) to gather completed sections and format them as a context for creating the final section"""

    return format_sections(completed_sections)

@task
def write_final_sections(
    section: Section, completed_report_sections: list[str]
) -> list[Section]:
    """
    Task (node) to write the final sections of the report.
    These sections do not require web searches and use the completed report sections as context
    """

    # Prepare the system instructions
    system_instructions = final_section_writer_instructions.format(
        section_title=section.name,
        section_topic=section.description,
        context=completed_report_sections,
    )

    # Generate the section content
    writer_model = ChatDatabricks(
        model=writer_model_name,
        temperature=0,
        max_tokens=MAX_WRITTING_TOKENS,
    )
    section_content = writer_model.invoke(
        [SystemMessage(content=system_instructions)]
        + [HumanMessage(content="Write the final sections of the report according to the provided information.")]
    )

    # Update the section content
    section.content = section_content.content

    # Return the updated section
    return section


@task
def compile_final_report(
    sections: list[Section], completed_sections: list[Section]
) -> str:
    """Task (node) to compile the final report"""

    completed_sections = {s.name: s.content for s in completed_sections}

    # Update the content of each section
    for section in sections:
        section.content = completed_sections[section.name]

    # Gather the sections into a single string
    all_sections = "\n\n".join([s.content for s in sections])

    return all_sections

@entrypoint(checkpointer=MemorySaver())
def deep_research_agent(inputs: dict):
    """Deep Research Agent to generate a report based on a research topic"""

    feedback_on_report_plan = None
    section_completed = False

    # Get the inputs
    completed_sections = inputs.get("completed_sections")
    sections = inputs.get("sections", completed_sections)

    research_topic = inputs.get("research_topic")
    if not research_topic:
        raise ValueError("research_topic is required")

    if not sections:
        while not section_completed:
            # generate search queries to help with planning the report
            queries = generate_report_plan(research_topic, feedback_on_report_plan)

            # search for the queries
            search_results = [
                search_query(
                    query, max_results=MAX_SEARCH_RESULT_FOR_SECTION_GENERATION
                )
                for query in queries.result()
            ]

            # generate the sections of the report
            sections = generate_sections(
                research_topic,
                search_results=[r.result() for r in search_results],
                feedback_on_report_plan=feedback_on_report_plan,
            ).result()

            # get feedback on the report plan
            section_completed, feedback_on_report_plan = human_feedback_for_sections(
                sections
            ).result()

    if not completed_sections:
        # build the completed sections
        build_results = [build_completed_section(section) for section in sections]
        completed_sections = [s.result() for s in build_results]

    # gather the completed sections
    report_sections_from_research = gather_completed_sections(
        completed_sections,
    ).result()

    # write the final sections
    additional_completed_sections = [
        write_final_sections(s, report_sections_from_research)
        for s in sections
        if not s.research
    ]
    completed_sections = completed_sections + [
        s.result() for s in additional_completed_sections
    ]

    # compile the final report
    final_report = compile_final_report(sections, completed_sections)

    return final_report.result()

# COMMAND ----------

import mlflow

mlflow.langchain.autolog()


# COMMAND ----------


from IPython.display import Markdown

def _print_deep_research_agent_step(agent, inputs, config) -> str:
    output_text = ""

    for event in agent.stream(inputs, config):
        for task_name, result in event.items():
            print(task_name)
            print("\n")

            if task_name in ("__interrupt__"):
                output_text = result[0].value
            elif task_name in ("deep_research_agent"):
                output_text = result

    return output_text

# COMMAND ----------
config = {"configurable": {"thread_id": "1"}}
research_topic = "What is AI Agent?"
inputs = {"research_topic": research_topic}

Markdown(
    _print_deep_research_agent_step(
        deep_research_agent,
        {"research_topic": research_topic},
        config,
    )
)


# COMMAND ----------

human_input = Command(resume="provide challenges and opportunities in AI Agent research")
Markdown(
    _print_deep_research_agent_step(
        deep_research_agent,
        human_input,
        config,
    )
)

# COMMAND ----------

human_input = Command(resume=True)
Markdown(
    _print_deep_research_agent_step(
        deep_research_agent,
        human_input,
        config,
    )
)
