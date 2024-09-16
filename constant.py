FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do, Strictly Use the any one of [{tool_names}] tools for first attempt.
If you need to use a tool to find the answer, use the following format:
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action,
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

"""

# Define the QA chain prompt
URL_RETRIEVAL_TEMPLATE = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
