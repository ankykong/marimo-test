import marimo

__generated_with = "0.11.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# **AI Agents** ⚒ Workshop by [DSML](https://www.instagram.com/dsmliui/)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Welcome to the AI Agents Workshop!""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this session, we will build an AI-powered resume evaluation tool. By the end of this workshop, you will:

        - Understand how AI agents work
        - Set up and use the **CrewAI** framework
        - Integrate **Gemini API** for AI-powered evaluation
        - Implement and test a resume evaluator

        Let's begin!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##  1: Install Required Libraries
        We will use [CrewAI](https://www.crewai.com) to build AI agents and Google's Gemini API for inference.

        **Instructions:**
        1. Run the cell below to install the required dependencies.
        2. Once installation is complete, **restart the runtime** when prompted.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2: Set Up Gemini API""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Google offers **$300 in free credits for 90 days** to use their Vertex AI API.

        **Steps to Set Up Your API Key:**

        1. Generate your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
        2. Create a _.env_ file
        3. Add in your API key to the file as so:

        `GOOGLE_API_KEY`=*your_api_key*

        Once set up, we can import CrewAI libraries.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3: Import Necessary Libraries""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now that we have installed the dependencies and set up API access, let’s import the required libraries.""")
    return


@app.cell
def _():
    # We will be Importing necessary libraries
    from crewai import Agent, LLM, Task, Crew
    from dotenv import load_dotenv
    import os

    load_dotenv()

    # Retrieve the API key from Colab Secrets
    api_key = os.getenv('GOOGLE_API_KEY')

    # This is a check if your api key is loaded
    print("API Key loaded:",api_key!=None)
    return Agent, Crew, LLM, Task, api_key, load_dotenv, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Initialize the Gemini Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now that we have the API key, set up your llm that you will be using. Here we are using **Gemini 2.0 Flash** to run all the queries.

        *Note 📝:*  **Gemini 2.0 Flash** is the latest iteration of Google's AI Models and is one of the fastest and best models available on the market
        """
    )
    return


@app.cell
def _(LLM, api_key):
    # Initialize the LLM with the Gemini API key
    my_llm = LLM(
       api_key=api_key,
       model="gemini/gemini-2.0-flash",
       verbose=True,
       temperature=0.5
    )
    return (my_llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4: Single Agent Example:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Before we move on with a multiple agent setup for this workshop. we will statrt by creating a simple agent and define its task.

        1.   This agent is a researcher that will research and provide information about latest AI advancements.
        """
    )
    return


@app.cell
def _(Agent, my_llm):
    # Define the agent
    researcher = Agent(
       name="Senior Researcher",
       role="Researcher",
       # Add the goal for the agent
       goal="To research and provide information about the latest AI advancements.",
       backstory="An expert in AI technologies with a focus on cutting-edge advancements.",
       llm=my_llm
    )
    return (researcher,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We now give the agent a single task to accomplish  
        1. Summarize latest advancements
        """
    )
    return


@app.cell
def _(Task, researcher):
    # Define the task
    task = Task(
       description="Summarize the latest advancements in AI from the last 6 months.",
       expected_output="A concise summary of recent AI developments.",
       agent=researcher
    )
    return (task,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Then we set up our crew of Agents. Here we only have one agent and one task.""")
    return


@app.cell
def _(Crew, researcher, task):
    # Create the Crew with the agent and task
    crew = Crew(
       agents=[researcher],
       tasks=[task]
    )
    return (crew,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Finally, we tell the Crew of 1 to start and we print out the final response we get.""")
    return


@app.cell
def _(mo):
    single_agent_start = mo.ui.run_button(label="Start Single Agent")
    single_agent_start
    return (single_agent_start,)


@app.cell
def _(crew, single_agent_start):
    # Start the process
    if single_agent_start.value:
        result = crew.kickoff()
        print(result)
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we have done this lets move forward start defining our Agents for our Resume Review and Interview preparation task.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5: Mutli-agent Example:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""First we need a pdf parser for our uploaded resume which will parse the text in the uploaded pdf.""")
    return


@app.cell
def _():
    # We will use pymupdf to do the parsing of the pdf, so we need to install it 
    import fitz  # PyMuPDF
    return (fitz,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Tools""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The first tool is for reading and extracting text from PDF files using the fitz library. It takes a file path as input, opens the PDF, extracts text from each page, and returns it. If no text is found or an error occurs, it provides an appropriate message.""")
    return


@app.cell
def _(fitz):
    from typing import Any, Optional, Type
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class PDFReaderToolSchema(BaseModel):
      """Input schema for PDFReaderTool """
      pdf_path :str = Field(..., description="Path to the pdf file")

    class PDFReaderTool(BaseTool):
      """Tool to read pdf files and extract text from it"""

      name: str = "pdf resume reader tool"
      description: str = "Tool to read resumepdf files and extract text from it"
      args_schema: Type[BaseModel] = PDFReaderToolSchema
      pdf_path: Optional[str] = None

      def __init__(self,
                   pdf_path: Optional[str] = None,
                   **kwargs: Any,
                   ):

        """
            Initializes the PDFReaderTool.

            Args:
                pdf_path (Optional[str]): Path to the PDF file.
        """
        super().__init__(**kwargs)
        if pdf_path is not None:
              self.pdf_path = pdf_path
              self.description = f"A tool that extracts text from {pdf_path}."
              self.args_schema = PDFReaderToolSchema

      def _run(self, **kwargs: Any) -> str:
        """
            Runs the PDFReaderTool.
            Args:
                **kwargs: Should contain 'pdf_path' as the file path.
            Returns:
                str: Extracted text from the PDF.

        """
        pdf_path = kwargs.get('pdf_path')
        if not pdf_path:
                return "❌ Error: No PDF path provided!"
        try:
          print(f" Extracting text from: {pdf_path}") # Print extracted text
          doc = fitz.open(pdf_path)
          text = "\n".join([page.get_text("text") for page in doc])
          return text if text else "No text found in the PDF."
        except Exception as e:
          return f"❌ Error: {str(e)}"


    pdf_reader_tool = PDFReaderTool()
    return (
        Any,
        BaseModel,
        BaseTool,
        Field,
        Optional,
        PDFReaderTool,
        PDFReaderToolSchema,
        Type,
        pdf_reader_tool,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We are also using a tool for webscraping , which is supported by CrewAI""")
    return


@app.cell
def _():
    from crewai_tools import ScrapeWebsiteTool

    scraper_tool = ScrapeWebsiteTool()
    return ScrapeWebsiteTool, scraper_tool


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### New Agents""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Agent 1: Resume Analyzer""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The first agent will extract information from the text of the resume.""")
    return


@app.cell
def _(Agent, my_llm):
    # Agent for extracting text from the resume

    resume_analyser = Agent(
        role="Resume Text Extractor",
        goal="Extract text from a resume document and summarize the key skills and experiences",
        backstory=(
            "You're a seasoned data researcher with a knack for finding the most relevant "
            "information. You're known for your ability to quickly scan through resumes "
            "and extract the most important details, making it easy for others to understand "
            "the key skills and experiences of a candidate."
        ),
        verbose=True,
        allow_delegation=False,
         llm=my_llm
    )
    return (resume_analyser,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Agent 2: Job Description Grabber""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The second agent will scrape websites using the tool to get the job description""")
    return


@app.cell
def _(Agent, my_llm):
    # This is our Second agent which is an expert in Extracting and presenting infromation from the given Job description.

    jobDesc_scrapper = Agent(
        role="Job Description Scraper",
        verbose=True,
        goal="Extract job details from the given job link and summarize the key requirements & skills in proper numbering format",
        backstory=(
            " You are a profession data reasearcher in the domian of finding the most relevant information"
            "You're known for your ability to quickly scan through the job description/postings and extract"
            "relevant details in proper number format, making it easy for others to understand what's required for a particular job role."),
        allow_delegation=False,
        llm = my_llm
    )
    return (jobDesc_scrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Agent 3: Custom Resume Writer""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Another agent creates a personalized resume using your resume and the job description.""")
    return


@app.cell
def _(Agent, my_llm):
    # This is a cv_writer agent - this wil write short note on why a candidate is suitable for a job.
    cv_writer = Agent(
        role="CV Writer",
        goal="Write a short note on why candidate is suitable for a job role",
        backstory=(
            "You are a professional CV writer. When provided with a candidate's resume and a job "
            "description, you can quickly summarize the candidate's key skills and experiences "
            "and write a personalized short note on why the candidate is suitable for the job role."
        ),
        verbose=True,
        allow_delegation=False,
         llm=my_llm
    )
    return (cv_writer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Agent 4: Interview Question Generator""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The final agent will craft interview questions for you to prepare using.""")
    return


@app.cell
def _(Agent, my_llm):
    interview_expert = Agent(
        role="Interview Expert",
        goal="Prepare a list of questions to ask a candidate based on their resume and job description",
        backstory=(
            "You are an experienced interviewer with a knack for asking the right questions. "
            "You can quickly scan through a candidate's resume and a job description and come up "
            "with a list of questions that might be asked during an interview."
        ),
        verbose=True,
        allow_delegation=False,
         llm=my_llm
    )
    return (interview_expert,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we have defined our agents lets define their task which they need to carryout.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Tasks""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Task for Resume Analyzer""")
    return


@app.cell
def _(Task, pdf_reader_tool, resume_analyser):
    resume_extract_task = Task(
        description="Extract text from a resume document available at {resume_path} and summarize the key skills and experiences.",
        expected_output="A summary of the key skills and experiences extracted from the resume",
        agent=resume_analyser,
        tools= [pdf_reader_tool]
    )
    return (resume_extract_task,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Task for Job Description Grabber""")
    return


@app.cell
def _(Task, jobDesc_scrapper, scraper_tool):
    jd_scrape_task = Task(
        description="Scrape job posting from a given URL {jd_url} and summarize the required skills.",
        expected_output="A summary of the job posting including required skills and responsibilities",
        agent=jobDesc_scrapper,
        tools= [scraper_tool]
    )
    return (jd_scrape_task,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Task for Custom Resume Writer""")
    return


@app.cell
def _(Task, cv_writer):
    cv_write_task = Task(
        description="Write a personalised short note on why a candidate is suitable for a job role.",
        expected_output="A short note highlighting the candidate's key skills and experiences and why they are suitable for the job role.",
        agent=cv_writer,
    )
    return (cv_write_task,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Task for Interview Question Generator""")
    return


@app.cell
def _(Task, interview_expert):
    interview_questions_task = Task(
        description="Prepare a list of questions to ask a candidate based on their resume and job description.",
        expected_output="A list of questions that might be asked during an interview based on the candidate's resume and job description.",
        agent=interview_expert,
    )
    return (interview_questions_task,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we have defined all the tasks for our agents. We now will set up our crew.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Set up your Crew""")
    return


@app.cell
def _(
    Crew,
    cv_write_task,
    cv_writer,
    interview_expert,
    interview_questions_task,
    jd_scrape_task,
    jobDesc_scrapper,
    resume_analyser,
    resume_extract_task,
):
    crew_1 = Crew(agents=[jobDesc_scrapper, resume_analyser, cv_writer, interview_expert], tasks=[jd_scrape_task, resume_extract_task, cv_write_task, interview_questions_task], verbose=True)
    return (crew_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Set your inputs

        *Instructions*

        1. Upload your resume to Colab
        2. Find a job that you are interested in and paste the URL here
        """
    )
    return


@app.cell
def _(mo):
    area=mo.ui.file(kind="area")
    button=mo.ui.file(kind="button")
    mo.vstack([button, area])
    return area, button


@app.cell
def _():
    import uuid
    return (uuid,)


@app.cell
def _(area, button, mo):
    if area.value is not None and len(area.value) > 0:
        uploaded_file = area.value[0]
    elif button.value is not None and len(button.value) > 0:
        uploaded_file = button.value[0]
    else:
        mo.md("No file uploaded yet.")
    file_name = uploaded_file.name
    file_content = uploaded_file.contents
    return file_content, file_name, uploaded_file


@app.cell
def _(file_content, file_name, mo, os, uuid):
    unique_filename = f"{uuid.uuid4()}_{file_name}"
    upload_dir = "uploads"  # Directory to store uploaded files
    os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists
    resume_path = os.path.join(upload_dir, unique_filename)
    # Save the file
    with open(resume_path, "wb") as f:
        f.write(file_content)
    mo.md(f"Uploaded file saved to: `{resume_path}`")
    return f, resume_path, unique_filename, upload_dir


@app.cell
def _(mo):
    jd = mo.ui.text_area(placeholder="https://www.google.com/about/careers/applications/jobs/results/110690555461018310-software-engineer-iii-infrastructure-core", label="Job Descriptoin URL")
    return (jd,)


@app.cell
def _(jd):
    jd
    return


@app.cell
def _(jd):
    jd.value
    return


@app.cell
def _(jd, resume_path):
    inputs = {
        "jd_url": str(jd.value),
        "resume_path": resume_path
    }
    return (inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Run Your Crew""")
    return


@app.cell
def _(mo):
    start_button = mo.ui.run_button(label="Run Crew")
    start_button
    return (start_button,)


@app.cell
def _(crew_1, inputs, start_button):
    if start_button.value:
        result_1 = crew_1.kickoff(inputs=inputs)
    return (result_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This will print out results of all the tasks from the Agents.""")
    return


@app.cell
def _(result_1):
    result_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We will print the tasks separately.""")
    return


@app.cell
def _(result_1):
    from IPython.display import Markdown
    Markdown(result_1.tasks_output[0].raw)
    return (Markdown,)


@app.cell
def _(Markdown, result_1):
    Markdown(result_1.tasks_output[1].raw)
    return


@app.cell
def _(Markdown, result_1):
    Markdown(result_1.tasks_output[2].raw)
    return


@app.cell
def _(Markdown, result_1):
    Markdown(result_1.tasks_output[3].raw)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# You can play around with all the agents. Try changing the behavior of the agents and their tasks. Utilize the Gemini api from Google for free.""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
