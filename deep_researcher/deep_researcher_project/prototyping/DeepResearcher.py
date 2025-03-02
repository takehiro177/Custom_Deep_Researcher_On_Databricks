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

# MAGIC %pip install -U databricks-sdk
# MAGIC %pip install langgraph databricks-langchain
# MAGIC %pip install "mlflow-skinny[databricks]>=2.20.2" loguru

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %%writefile prompts.py
# MAGIC
# MAGIC # Prompt to generate search queries to help with planning the report
# MAGIC report_planner_query_writer_instructions="""You are performing research for a report. 
# MAGIC
# MAGIC <Report topic>
# MAGIC {topic}
# MAGIC </Report topic>
# MAGIC
# MAGIC <Report organization>
# MAGIC {report_organization}
# MAGIC </Report organization>
# MAGIC
# MAGIC <Feedback>
# MAGIC {feedback}
# MAGIC </Feedback>
# MAGIC
# MAGIC <Task>
# MAGIC Your goal is to generate {number_of_queries} web search queries that will help gather information for planning the report sections. 
# MAGIC
# MAGIC The queries should:
# MAGIC
# MAGIC 1. Be related to the Report topic
# MAGIC 2. Help satisfy the requirements specified in the report organization
# MAGIC 3. If feedback is populated, it should be added to enhance the report queries.
# MAGIC
# MAGIC Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
# MAGIC </Task>
# MAGIC """
# MAGIC
# MAGIC # Prompt to generate the report plan
# MAGIC report_planner_instructions="""I want a plan for a report that is concise and focused.
# MAGIC
# MAGIC <Task>
# MAGIC Generate a list of sections for the report. Your plan should be tight and focused with NO overlapping sections or unnecessary filler. 
# MAGIC
# MAGIC Each section should have the fields:
# MAGIC
# MAGIC - Name - Name for this section of the report.
# MAGIC - Description - Brief overview of the main topics covered in this section.
# MAGIC - Research - Whether to perform web research for this section of the report.
# MAGIC - Content - The content of the section, which you will leave blank for now.
# MAGIC
# MAGIC Integration guidelines:
# MAGIC - Include examples and implementation details within main topic sections, not as separate sections
# MAGIC - Ensure each section has a distinct purpose with no content overlap
# MAGIC - Combine related concepts rather than separating them
# MAGIC
# MAGIC Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow. You must follow following format strictly.
# MAGIC </Task>
# MAGIC
# MAGIC <format>
# MAGIC class Section(BaseModel):
# MAGIC     name: str = Field(
# MAGIC         description="Name for this section of the report.",
# MAGIC     )
# MAGIC     description: str = Field(
# MAGIC         description="Brief overview of the main topics and concepts to be covered in this section.",
# MAGIC     )
# MAGIC     research: bool = Field(
# MAGIC         description="Whether to perform web research for this section of the report."
# MAGIC     )
# MAGIC     content: str = Field(description="The content of the section.")
# MAGIC
# MAGIC class Sections(BaseModel):
# MAGIC     sections: List[Section] = Field(
# MAGIC         description="Sections of the report.",
# MAGIC     )
# MAGIC </format>
# MAGIC
# MAGIC <Report topic>
# MAGIC The topic of the report is:
# MAGIC {topic}
# MAGIC </Report topic>
# MAGIC
# MAGIC <Report organization>
# MAGIC The report should follow this organization: 
# MAGIC {report_organization}
# MAGIC </Report organization>
# MAGIC
# MAGIC <Context>
# MAGIC Here is context to use to plan the sections of the report: 
# MAGIC {context}
# MAGIC </Context>
# MAGIC
# MAGIC <Feedback>
# MAGIC Here is feedback on the report structure from review (if any):
# MAGIC {feedback}
# MAGIC </Feedback>
# MAGIC """
# MAGIC
# MAGIC # Query writer instructions
# MAGIC query_writer_instructions="""You are an expert technical writer crafting targeted web search queries that will gather comprehensive information for writing a technical report section.
# MAGIC
# MAGIC <Report topic>
# MAGIC {topic}
# MAGIC </Report topic>
# MAGIC
# MAGIC <Section topic>
# MAGIC {section_topic}
# MAGIC </Section topic>
# MAGIC
# MAGIC <Task>
# MAGIC Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information above the section topic. 
# MAGIC
# MAGIC The queries should:
# MAGIC
# MAGIC 1. Be related to the topic 
# MAGIC 2. Examine different aspects of the topic
# MAGIC
# MAGIC Make the queries specific enough to find high-quality, relevant sources.
# MAGIC
# MAGIC You must return only a list of {number_of_queries} search queries in the format below.
# MAGIC </Task>
# MAGIC """
# MAGIC
# MAGIC # Section writer instructions
# MAGIC section_writer_instructions = """You are an expert technical writer crafting one section of a technical report.
# MAGIC
# MAGIC <Report topic>
# MAGIC {topic}
# MAGIC </Report topic>
# MAGIC
# MAGIC <Section name>
# MAGIC {section_name}
# MAGIC </Section name>
# MAGIC
# MAGIC <Section topic>
# MAGIC {section_topic}
# MAGIC </Section topic>
# MAGIC
# MAGIC <Existing section content (if populated)>
# MAGIC {section_content}
# MAGIC </Existing section content>
# MAGIC
# MAGIC <Source material>
# MAGIC {context}
# MAGIC </Source material>
# MAGIC
# MAGIC <Guidelines for writing>
# MAGIC 1. If the existing section content is not populated, write a new section from scratch.
# MAGIC 2. If the existing section content is populated, write a new section that synthesizes the existing section content with the Source material.
# MAGIC </Guidelines for writing>
# MAGIC
# MAGIC <Length and style>
# MAGIC - Strict 350-500 word limit
# MAGIC - No marketing language
# MAGIC - Technical focus
# MAGIC - Write in simple, clear language
# MAGIC - Start with your most important insight in **bold**
# MAGIC - Use short paragraphs (2-5 sentences max)
# MAGIC - Use ## for section title (Markdown format)
# MAGIC - Only use ONE structural element IF it helps clarify your point:
# MAGIC   * Either a focused table comparing 2-3 key items (using Markdown table syntax)
# MAGIC   * Or a short list (3-5 items) using proper Markdown list syntax:
# MAGIC     - Use `*` or `-` for unordered lists
# MAGIC     - Use `1.` for ordered lists
# MAGIC     - Ensure proper indentation and spacing
# MAGIC - End with ### Sources that references the below source material formatted as:
# MAGIC   * List each source with title, date, and URL
# MAGIC   * Format: `- Title : URL`
# MAGIC </Length and style>
# MAGIC
# MAGIC <Quality checks>
# MAGIC - Exactly 350-500 words (excluding title and sources)
# MAGIC - Careful use of only ONE structural element (table or list) and only if it helps clarify your point
# MAGIC - One specific example / case study
# MAGIC - Starts with bold insight
# MAGIC - No preamble prior to creating the section content
# MAGIC - Sources cited at end
# MAGIC </Quality checks>
# MAGIC """
# MAGIC
# MAGIC # Instructions for section grading
# MAGIC section_grader_instructions = """Review a report section relative to the specified topic:
# MAGIC
# MAGIC <Report topic>
# MAGIC {topic}
# MAGIC </Report topic>
# MAGIC
# MAGIC <section topic>
# MAGIC {section_topic}
# MAGIC </section topic>
# MAGIC
# MAGIC <section content>
# MAGIC {section}
# MAGIC </section content>
# MAGIC
# MAGIC <task>
# MAGIC Evaluate whether the section content adequately addresses the section topic.
# MAGIC
# MAGIC If the section content does not adequately address the section topic, generate follow-up search queries to gather missing information.
# MAGIC </task>
# MAGIC
# MAGIC <format>
# MAGIC     grade: Literal["pass","fail"] = Field(
# MAGIC         description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
# MAGIC     )
# MAGIC     follow_up_queries: List[SearchQuery] = Field(
# MAGIC         description="List of follow-up search queries.",
# MAGIC     )
# MAGIC </format>
# MAGIC """
# MAGIC
# MAGIC final_section_writer_instructions="""You are an expert technical writer crafting a section that synthesizes information from the rest of the report.
# MAGIC
# MAGIC <Report topic>
# MAGIC {topic}
# MAGIC </Report topic>
# MAGIC
# MAGIC <Section name>
# MAGIC {section_name}
# MAGIC </Section name>
# MAGIC
# MAGIC <Section topic> 
# MAGIC {section_topic}
# MAGIC </Section topic>
# MAGIC
# MAGIC <Available report content>
# MAGIC {context}
# MAGIC </Available report content>
# MAGIC
# MAGIC <Task>
# MAGIC 1. Section-Specific Approach:
# MAGIC
# MAGIC For Introduction:
# MAGIC - Use # for report title (Markdown format)
# MAGIC - 50-100 word limit
# MAGIC - Write in simple and clear language
# MAGIC - Focus on the core motivation for the report in 1-2 paragraphs
# MAGIC - Use a clear narrative arc to introduce the report
# MAGIC - Include NO structural elements (no lists or tables)
# MAGIC - No sources section needed
# MAGIC
# MAGIC For Conclusion/Summary:
# MAGIC - Use ## for section title (Markdown format)
# MAGIC - 100-150 word limit
# MAGIC - For comparative reports:
# MAGIC     * Must include a focused comparison table using Markdown table syntax
# MAGIC     * Table should distill insights from the report
# MAGIC     * Keep table entries clear and concise
# MAGIC - For non-comparative reports: 
# MAGIC     * Only use ONE structural element IF it helps distill the points made in the report:
# MAGIC     * Either a focused table comparing items present in the report (using Markdown table syntax)
# MAGIC     * Or a short list using proper Markdown list syntax:
# MAGIC       - Use `*` or `-` for unordered lists
# MAGIC       - Use `1.` for ordered lists
# MAGIC       - Ensure proper indentation and spacing
# MAGIC - End with specific next steps or implications
# MAGIC - No sources section needed
# MAGIC
# MAGIC 3. Writing Approach:
# MAGIC - Use concrete details over general statements
# MAGIC - Make every word count
# MAGIC - Focus on your single most important point
# MAGIC </Task>
# MAGIC
# MAGIC <Quality Checks>
# MAGIC - For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
# MAGIC - For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
# MAGIC - Markdown format
# MAGIC - Do not include word count or any preamble in your response
# MAGIC </Quality Checks>"""
# MAGIC
# MAGIC

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

def deduplicate_and_format_sources(search_responses, max_tokens_per_source, include_raw_content=True):
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

# MAGIC %sql
# MAGIC CREATE CONNECTION tavily_api TYPE HTTP
# MAGIC OPTIONS (
# MAGIC   host 'https://api.tavily.com',
# MAGIC   port '443',
# MAGIC   base_path '/',
# MAGIC   bearer_token secret ('tk-personal','dev-tavily')
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION dev.deep_researcher.tavily_web_search(
# MAGIC   query STRING COMMENT 'query text'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Performs a web search using Tavily'
# MAGIC RETURN (SELECT http_request(
# MAGIC   conn => 'tavily_api',
# MAGIC   method => 'POST',
# MAGIC   path => 'search',
# MAGIC   json => to_json(named_struct(
# MAGIC     'query', query
# MAGIC   )),
# MAGIC   headers => map(
# MAGIC     'Content-Type', "application/json"
# MAGIC   )
# MAGIC )).text

# COMMAND ----------

from typing import Literal
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from databricks_langchain import ChatDatabricks

from loguru import logger

writer_model_name = "databricks-meta-llama-3-3-70b-instruct"
planner_model_name = "databricks-meta-llama-3-3-70b-instruct" #"deepseek_r1_distilled_llama8b_v1"

MAX_SEARCH_RESULT_FOR_SECTION_GENERATION = 3
MAX_SEARCH_RESULT_FOR_SECTION_COMPLETION = 3
MAX_LOOP_FOR_BUILD_SECTION = 1
MAX_TOKENS_PER_SOURCE = 1000
MAX_WRITTING_TOKENS = 4000
MAX_PLANNING_TOKENS = 4000

from prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions,
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
)

DEFAULT_REPORT_STRUCTURE = None

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
        topic=section.name,
        section_name=section.name,
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
        topic=section.name,
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
    content="Detail of AI Agent for beginners",
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
  #  source_str = deduplicate_and_format_sources(
  #      search_results,
  #      max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
  #      include_raw_content=False,
  #  )

    # Prepare the system prompt
    system_instructions_sections = report_planner_instructions.format(
        topic=topic,
        report_organization=report_structure,
        context=search_results,
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
                    The response must include a 'sections' field containing a list of a section. \
                    Each section must include name, description, plan, research, and content fields. Do not include any other fields."
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
    #sections_str = "\n\n".join(
    #    f"Section: {section.name}\n"
    #    f"Description: {section.description}\n"
    #    f"Research needed: {'Yes' if section.research else 'No'}\n"
    #    for section in sections
    #)

    # Get feedback on the report plan
    feedback = interrupt(
       f"Please provide feedback on the report plan below.\n\nDoes the report plan meet your needs? Pass 'true' to approve the report plan or provide feedback to regenerate the report plan." #f"Please provide feedback on the report plan below.\n\n{sections_str}\n\nDoes the report plan meet your needs? Pass 'true' to approve the report plan or provide feedback to regenerate the report plan."
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
            "max_loops_for_build_section": MAX_LOOP_FOR_BUILD_SECTION,
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
        topic=section.name,
        section_name=section.name,
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
        sections = generate_sections(
                research_topic,
                search_results=[r.result() for r in search_results],
                feedback_on_report_plan=feedback_on_report_plan,
            ).result()
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
        inputs,
        config,
    )
)


# COMMAND ----------

human_input = Command(resume="provide challenges and opportunities of research in AI Agent")
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