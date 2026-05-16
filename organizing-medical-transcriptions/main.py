import os
import pandas as pd
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Load the data
df = pd.read_csv("data/transcriptions.csv")
result_list = []

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

for _, row in df.iterrows():
    row_json = row.to_json(orient='records')
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "you are a data analyst."},
            {"role": "user", "content": f"analyze this data: {row_json}"}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "extract_needed_info",
                "description": "get the patient information from the json input of the body and give the recommended treatment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "string", "description": "age of the patient"},
                        "medical_specialty": {"type": "string", "description": "medical specialty"},
                        "recommended_treatment": {"type": "string", "description": "recommended treatment"},
                        "ICD_code": {"type": "string", "description": "ICD code corresponding to the transcript"}
                    }
                }
            }
        }]
    )
    
    message = response.choices[0].message
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        arguments = getattr(getattr(tool_call, "function", None), "arguments", None)
        if arguments is None:
            arguments = getattr(tool_call, "arguments", "{}")
        result_list.append(json.loads(arguments))

df_structured = pd.DataFrame(result_list)
print(df_structured)