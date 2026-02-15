# LangChain Course

Este repositório contém scripts e exemplos de código para aprendizado de LangChain com Python, integrado com Google Gemini.

## Pré-requisitos

- Python 3.13+
- Chave de API do Google AI Studio (`GOOGLE_API_KEY`)

## Instalação

As dependências são gerenciadas via `uv` ou `pip`. O arquivo `pyproject.toml` contém as definições.

```bash
pip install -r requirements.txt
# ou se estiver usando uv:
uv sync
```

## Estrutura do Projeto

- `main.py`: Script principal demonstrando um agente ReAct simples usando Gemini e pesquisa na web.
- `list_models.py`: Script utilitário para listar modelos disponíveis na sua conta Google AI.
- `test_langchain_direct.py`: Script de teste para validação básica.

## Configuração

Crie um arquivo `.env` na raiz do projeto com sua chave de API:

```env
GOOGLE_API_KEY=sua_chave_aqui
```

## Execução

Para rodar o agente principal:

```bash
python main.py
```

Para verificar modelos disponíveis:

```bash
python list_models.py
```
