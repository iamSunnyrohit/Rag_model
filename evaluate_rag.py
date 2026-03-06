from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Setup the "Judge" (We'll use Llama 3.2 as the grader)
judge_llm = ChatOllama(model="llama3.2", temperature=0)

# 2. The Grading Prompt
eval_prompt = ChatPromptTemplate.from_template("""
You are an expert grader. Compare the STUDENT ANSWER to the REFERENCE CONTEXT.
Give a score from 1 to 10 for 'Faithfulness' (10 means no hallucinations).

REFERENCE CONTEXT:
{context}

STUDENT ANSWER:
{answer}

Provide the score and a one-sentence reason.
""")

# 3. Test Data (From your previous successful run)
context = "To download a dataset: kaggle datasets download dataset_name"
llama_answer = "Use 'kaggle datasets download dataset_name' to get the files."

# 4. Run Evaluation
print("⚖️ Grading Llama 3.2's performance...")
chain = eval_prompt | judge_llm | StrOutputParser()
result = chain.invoke({"context": context, "answer": llama_answer})

print(f"\n--- EVALUATION REPORT ---\n{result}")