import os
import json
import azure.functions as func
import logging
import time
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
#from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

app = func.FunctionApp(http_auth_level=func.AuthLevel.ADMIN)

# Load prompt from file
def load_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# base model for selections
class Selection(BaseModel):
    Selection: str = Field(
        ..., description="The selection of the key-value pair", example="selection"
        )

# base model for key value pairs
class KeyValuePair(BaseModel):
    question_number: Optional[str] = Field(
        None, description="Question number from the original document", example="1"
        )
    key: str = Field(
        ..., description="The key of the key-value pair", example="key"
        )
    value: str = Field(
        ..., description="The value of the key-value pair", example="value"
        )
    notes: Optional[str] = Field(
        None, description="Notes in the document for the key-value pair", example="notes"
        )
# base model for CheckListSubSection
class ChecklistSubSection(BaseModel):
    title: str = Field(
        ..., description="The title of the checklist sub-section", example="sub-section title"
        )
    items: List[KeyValuePair] = Field(
        ..., description="The items in the checklist sub-section", example=[{"key": "key", "value": "value"}]
        )

# base model for Checklist sections
class ChecklistSection(BaseModel):
    title: str = Field(
        ..., description="The title of the checklist section", example="section title"
        )
    summary: str = Field(
        ..., description="Summarize the data included in the section. Create narrative", example="section summary"
        )
    # items: List[KeyValuePair] = Field(
    #     ..., description="The items in the checklist section", example=[{"key": "key", "value": "value"}]
    #     )
    subsections: List[ChecklistSubSection] = Field(
        ..., description="The subsections in the checklist section", example=[{"title": "sub-section title", "items": [{"key": "key", "value": "value"}]}]
        )


# Model for the entire Checklist
class Checklist(BaseModel):
    summary: str = Field(
        ..., description="Narrative summary of the overall data in the document.", example="checklist summary"
        )
    sections: List[ChecklistSection] = Field(
        ..., description="The sections of the checklist", example=[
            {
                "title": "section title 1",
                "items": [{"number":"1", "key": "key1", "value": "value1", "notes": "notes"}]
            },
            {
                "title": "section title 2",
                "items": [{"number":"1", "key": "key2", "value": "value2", "notes": "notes"}]
            }
        ]
    )    

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    file_path = "./example.pdf"
    endpoint = os.getenv('AZURE_COGNITIVE_ENDPOINT')
    key = os.getenv('AZURE_COGNITIVE_KEY')
    loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=endpoint,
        api_key=key,
        file_path=file_path,
        api_model="prebuilt-layout"
    )

    logging.info(f"Loader Initialized: {loader}")
    document = loader.load()
    logging.info(f"Document Loaded: {document}")

    # Send the PDF file directly to Azure OpenAI


    # Initialize Azure OpenAI
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        temperature=0,
        top_p=1.0,
        verbose=True,
        azure_endpoint=os.getenv('AZURE_OPENAI_API_BASE')
    )
    logging.info(f"LLM Initialized: {llm}")

    promptTemplate = load_prompt("./prompt/prompt.txt")
    logging.info(f"Prompt Template: {promptTemplate}")

    prompt = ChatPromptTemplate.from_template(promptTemplate)
    structured_output = llm.with_structured_output(Checklist)

    results = structured_output.invoke(
        prompt.invoke(
            {
                "input": document
            }
        )
    )

    # Start timing
    start_time = time.time()

    summaryPromptTemplate = load_prompt("./prompt/summary.txt")
    summaryPrompt = ChatPromptTemplate.from_template(summaryPromptTemplate)
    summary = llm.invoke(summaryPrompt.invoke({"input": results}))

    # End timing
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Log the elapsed time
    logging.info(f"Time taken to generate summary: {elapsed_time:.2f} seconds")

    # append summary to results json
    logging.info(f"[summary: {summary}]")
    results.summary = summary.content

    # Save formatted JSON to file for Demo
    results_dict = results.dict()
    formatted_results = json.dumps(results_dict, indent=4)
    with open("results.json", "w") as file:
        file.write(formatted_results)


    # return structured response JSON
    return func.HttpResponse(
        results.json(), 
        status_code=200, 
        mimetype="application/json"
    )