from pathlib import Path
from pprint import pprint
import asyncio
from datetime import datetime

import fire
from synago.agent import Agent
from synago.tools.duckduckgo import duckduckgo_search
from synago.tools.web_crawl import web_crawl
from loguru import logger
from pydantic import BaseModel, Field


HERE = Path(__file__).parent


default_theme = "The applications of LLM-based agents in biology and medicine."


async def generate_report(
    theme: str = default_theme,
    output: str | None = None,
    results_per_keyword: int = 5,
):
    query_keywords_agent = Agent(
        name="query_keywords_agent",
        instructions="""You are a search engine expert,
    you can generate a list of query keywords for a search engine to find the most relevant papers.

    ## Duckduckgo query operators

    | Keywords example |	Result|
    | ---     | ---   |
    | cats dogs |	Results about cats or dogs |
    | "cats and dogs" |	Results for exact term "cats and dogs". If no results are found, related results are shown. |
    | cats -dogs |	Fewer dogs in results |
    | cats +dogs |	More dogs in results |
    | cats filetype:pdf |	PDFs about cats. Supported file types: pdf, doc(x), xls(x), ppt(x), html |
    | dogs site:example.com  |	Pages about dogs from example.com |
    | cats -site:example.com |	Pages about cats, excluding example.com |
    | intitle:dogs |	Page title includes the word "dogs" |
    | inurl:cats  |	Page url includes the word "cats" |
    """,
        model="gpt-4o-mini",
    )

    def merge_search_results(results: list[dict]) -> list[dict]:
        _dict = {}
        for result in results:
            _dict[result["title"]] = result
        return list(_dict.values())

    info_extraction_agent = Agent(
        name="info_extraction_agent",
        instructions=f"""You are a expert in the theme: `{theme}`,
    you should extract the paper title, summary, journal, time from the page content.
    You should also check if the search result is a paper and related to the theme.

    Please be very strict and careful,
    only return True if the paper is very related to the theme.
    """,
        model="gpt-4o-mini",
    )

    format_agent = Agent(
        name="format_agent",
        instructions=f"""You are a format agent,
    you should format the answer of other agent give a markdown format.
    List all the papers to markdown points.

    Add a well-formatted title and a descriptions about the theme `{theme}`.
    """,
        model="gpt-4o-mini",
    )

    class QueryKeywords(BaseModel):
        keywords: list[str]

    query_keywords = await query_keywords_agent.run(
        "Papers about applications of LLM-based agents in biology and medicine",
        response_format=QueryKeywords,
    )

    logger.info("Query keywords:")
    pprint(query_keywords.content.keywords)

    search_results = []
    for keyword in query_keywords.content.keywords:
        try:
            results = duckduckgo_search(keyword, max_results=results_per_keyword)
            await asyncio.sleep(1)
            search_results.extend(results)
        except Exception as e:
            logger.error(e)
    merged_results = merge_search_results(search_results)

    logger.info(f"Number of items before relation check: {len(merged_results)}")

    contents = await web_crawl([result["href"] for result in merged_results])

    class ContentInfo(BaseModel):
        title: str
        url: str
        summary: str
        is_related: bool = Field(description="Whether the paper is related to the theme")
        is_a_paper: bool = Field(description="Whether the content is a journal or preprint paper")
        journal: str = Field(description="The journal name of the paper")
        time: str = Field(description="The time of the paper")

    async def process_content(content, result):
        try:
            resp = await info_extraction_agent.run(
                result["href"] + "\n" + content, response_format=ContentInfo)
            logger.info(resp.content)
            if resp.content.is_related and resp.content.is_a_paper:
                return resp.content
        except Exception as e:
            logger.error(e)
        return None

    tasks = [process_content(content, result) 
             for content, result in zip(contents, merged_results)]
    results = await asyncio.gather(*tasks)
    list_of_info = [r for r in results if r is not None]

    logger.info(f"Number of items after relation check: {len(list_of_info)}")

    markdown = await format_agent.run(list_of_info)
    logger.info("Markdown:")
    print(markdown.content)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(markdown.content)


async def compare_and_update(target_file: str, increment_file: str, output_file: str):
    compare_agent = Agent(
        name="compare_agent",
        instructions="""You are a compare agent,
you should compare the paper list in the target file and the increment file,
and update the target file if the increment file is newer,
but you should keep the format of the original file.
And the paper list should be sorted by the time.
""",
        model="gpt-4o",
    )

    with open(target_file, "r", encoding="utf-8") as f:
        target_content = f.read()
    with open(increment_file, "r", encoding="utf-8") as f:
        increment_content = f.read()

    class CompareInput(BaseModel):
        target_content: str
        increment_content: str

    compare_input = CompareInput(
        target_content=target_content,
        increment_content=increment_content
    )

    class CompareOutput(BaseModel):
        is_updated: bool = Field(description="Whether the target file is updated")
        new_markdown: str = Field(description="The new markdown content")
        number_of_new_papers: int = Field(description="The number of new papers")
        new_papers: list[str] = Field(description="The new papers")

    resp = await compare_agent.run(compare_input, response_format=CompareOutput)
    logger.info("Compare result:")
    print("Is updated:", resp.content.is_updated)
    print("Number of new papers:", resp.content.number_of_new_papers)
    print("New papers:")
    pprint(resp.content.new_papers)

    if resp.content.is_updated:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(resp.content.new_markdown)


async def update_readme():
    # make the dir for store the daily report
    root_dir = (HERE / "..").absolute()
    daily_reports_dir = root_dir / "daily_reports"
    daily_reports_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")

    # generate the daily report
    await generate_report(
        theme="The applications of LLM-based agents in biology and medicine",
        output=daily_reports_dir / f"{today}.md",
    )

    # compare the daily report with the previous report
    await compare_and_update(
        target_file=root_dir / "README.md",
        increment_file=daily_reports_dir / f"{today}.md",
        output_file=root_dir / "README.md",
    )


if __name__ == "__main__":
    fire.Fire({
        "update_readme": update_readme,
        "generate_report": generate_report,
        "compare_and_update": compare_and_update,
    })
