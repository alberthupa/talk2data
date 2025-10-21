# from skeleton_codes.llms.basic_agent import BasicAgent
from basic_agent import BasicAgent

llm_agent = BasicAgent()

ai_response = llm_agent.get_text_response_from_llm(
    # llm_model_input="gemini-2.0-flash-exp",
    # llm_model_input="priv_openai:gpt-5",
    llm_model_input="lingaro:gpt-4o-mini",
    messages="hi",
    # code_tag=None,
)

print(ai_response)
