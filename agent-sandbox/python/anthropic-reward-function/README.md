# Buildfunctions SDK Examples

<p align="center">
  <h1 align="center">
  <a href="https://www.buildfunctions.com" target="_blank">
    <img src="../../../public/readme/buildfunctions-header.svg" alt="logo" width="900">
  </a>
  </h1>
</p>

<h1 align="center">Buildfunctions: Serverless functions for AI</h1>

<p align="center">
  <a href="https://discord.com/users/buildfunctions" target="_blank">
    <img src="../../../public/readme/discord-button.png" height="32" />
  </a>&nbsp;
  <a href="https://www.buildfunctions.com/docs/company/buildfunctions" target="_blank">
    <img src="../../../public/readme/read-the-docs-button.png" height="32" />
  </a>&nbsp;
</p>

<p align="center">
<a href="https://www.npmjs.com/package/buildfunctions" target="_blank">
  <img src="https://img.shields.io/badge/npm-@buildfunctions-green">
</a>
<a href="https://pypi.org/project/buildfunctions/" target="_blank">
  <img src="https://img.shields.io/badge/pip-buildfunctions-blue">
</a>
</p>

<p align="center">
  <h1 align="center">
  <a href="https://www.buildfunctions.com" target="_blank">
    <img src="../../../public/readme/buildfunctions-logo-and-servers-dark.svg" alt="logo" width="900">
  </a>
  </h1>
</p>

> **The Buildfunctions SDK for AI Agents** - Hardware-isolated CPU and GPU Sandboxes for untrusted AI actions.

## Installation

```bash
# TypeScript / JavaScript
npm install buildfunctions

# Python
pip install buildfunctions
```

## Examples

### Anthropic: Code Generation with Reward Scoring

Uses Claude to generate a sorting function, then executes and scores it inside an isolated CPU sandbox with a reward function that checks correctness and edge case handling.

**Files:**
- [`basic.py`](./basic.py) - Main example: generate code with Claude, evaluate in a sandbox
- [`reward_handler.py`](./reward_handler.py) - Reward function template that scores generated code
- [`requirements.txt`](./requirements.txt) - Python dependencies

**How it works:**
1. Authenticate with Buildfunctions
2. Generate a sorting function using Claude Sonnet
3. Create a CPU Sandbox with a reward handler that tests the generated code
4. Run the sandbox and get a reward score
5. Clean up the sandbox

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
```

`.env` requires:
```
BUILDFUNCTIONS_API_TOKEN=""
ANTHROPIC_KEY=""
```

## Run

```bash
# Run 
python basic.py
```