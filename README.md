# Buildfunctions SDK Examples

<p align="center">
  <h1 align="center">
  <a href="https://www.buildfunctions.com" target="_blank">
    <img src="./public/readme/buildfunctions-header.svg" alt="logo" width="900">
  </a>
  </h1>
</p>

<h1 align="center">Buildfunctions: Serverless functions for AI</h1>

<p align="center">
  <a href="https://discord.com/users/buildfunctions" target="_blank">
    <img src="./public/readme/discord-button.png" height="32" />
  </a>&nbsp;
  <a href="https://www.buildfunctions.com/docs/company/buildfunctions" target="_blank">
    <img src="./public/readme/read-the-docs-button.png" height="32" />
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
    <img src="./public/readme/buildfunctions-logo-and-servers-dark.svg" alt="logo" width="900">
  </a>
  </h1>
</p>

> **The Buildfunctions SDK for AI Agents** - Hardware-isolated CPU and GPU Sandboxes for untrusted AI actions.

## Installation

```bash
# TypeScript / JavaScript
npm install buildfunctions
# or
yarn add buildfunctions
# or
pnpm add buildfunctions

# Python
pip install buildfunctions
```

## Example

### Anthropic: Code Generation with Reward Scoring

Uses Claude to generate a sorting function, then executes and scores it inside an isolated CPU sandbox with a reward function that checks correctness and edge case handling.

**Files:**
- [`python/anthropic/agent-sandbox-reward-function/main.py`](./python/anthropic/agent-sandbox-reward-function/main.py) - Main example: generate code with Claude, evaluate in a sandbox
- [`python/anthropic/agent-sandbox-reward-function/reward_handler.py`](./python/anthropic/agent-sandbox-reward-function/reward_handler.py) - Reward function template that scores generated code
- [`python/anthropic/agent-sandbox-reward-function/requirements.txt`](./python/anthropic/agent-sandbox-reward-function/requirements.txt) - Python dependencies

**How it works:**
1. Authenticate with Buildfunctions
2. Generate a sorting function using Claude Sonnet
3. Create a CPU Sandbox with a reward handler that tests the generated code
4. Run the sandbox and get a reward score
5. Clean up the sandbox

## A Python Example

### Download

```bash
git clone https://github.com/buildfunctions/examples.git
cd examples/python/anthropic/agent-sandbox-reward-function
```
### Install dependencies

```bash
pip install -r requirements.txt
```

### Update environment variables

```bash
# Create an API token at https://www.buildfunctions.com/settings
cp .env.example .env
```

`.env` requires:
```
BUILDFUNCTIONS_API_TOKEN=""
ANTHROPIC_KEY=""
```

### Run

```bash
python main.py
```