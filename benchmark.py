import time
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

def run_benchmark(model_name, query):
    llm = ChatOllama(model=model_name, temperature=0)
    chain = llm | StrOutputParser()
    
    start_time = time.time()
    response = chain.invoke(query)
    end_time = time.time()
    
    duration = end_time - start_time
    word_count = len(response.split())
    tokens = word_count * 1.3
    tps = tokens / duration
    
    return {
        "model": model_name,
        "time": round(duration, 2),
        "tps": round(tps, 2),
        "response": response[:100] + "..."
    }

query = "Explain the importance of data cleaning in Machine Learning."
models = ["llama3.2", "mistral"]

print("🚀 Starting Benchmark on Mac mini...")
for model in models:
    print(f"Testing {model}...")
    stats = run_benchmark(model, query)
    print(f"✅ {stats['model']}: {stats['time']}s | {stats['tps']} tokens/sec")
    print("-" * 30)