import torch
print(torch.cuda.is_available())  # Should print False


from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
ddg_search = DuckDuckGoSearchAPIWrapper()

print(ddg_search)
