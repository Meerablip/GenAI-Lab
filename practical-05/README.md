# Practical 5 - Agentic Text-to-SQL

Natural-language business questions -> SQL (LLM + schema) -> read-only validation -> execution -> error feedback and
revision (max 2) -> result analysis -> structured JSON report. SQLite sales database (customers, products, orders,
order_items), `qwen2.5-coder:7b` served by Ollama on a GitHub-hosted CPU runner.

* `text2sql/database.py` - seeded database and schema extraction
* `text2sql/questions.json` - 12 questions with gold SQL + 2 unsafe requests (delete / drop)
* `text2sql/run_experiment.py` - the agent, evaluation (execution accuracy vs. gold) and reports
* Workflow: `.github/workflows/practical-05.yml`
