# RAG相关记录

## Code

1. [创建向量数据库](https://github.com/InternLM/Tutorial/blob/main/langchain/demo/create_db.py)
2. [根据实战营tutorial](https://github.com/InternLM/Tutorial/blob/main/helloworld/hello_world.md#25-web-demo-%E8%BF%90%E8%A1%8C)以及[往届项目参考](https://github.com/lindsey-chang/TRLLM-Traffic-Rules-Assistant/blob/main/web_demo_ensemble_retriever.py)运行demo.py

注：相关路径根据自身情况更改，将demo.py放在web_demo.py同级目录，运行streamlit记得更改文件

## 总结

1. 创建向量数据库对分块敏感
2. 对prompt设计需要更加精妙
3. RAG没有预想的效果好，预计将RAG文本通过API转换为QA进行微调
