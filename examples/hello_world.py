"""Hello World: Simplest possible DAG example."""

import asyncio

from nano_agent import DAG, ClaudeAPI


async def main() -> None:
    dag = (
        DAG().system("You are a friendly assistant.").user("Hello! What is your name?")
    )
    api = ClaudeAPI()
    response = await api.send(dag)
    dag = dag.assistant(response.content)
    print(dag)


if __name__ == "__main__":
    asyncio.run(main())
