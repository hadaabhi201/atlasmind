# AtlasMind: Multi-Tool Knowledge and Reasoning Engine

**AtlasMind** is a **LlamaIndex-powered multimodal reasoning agent** that integrates a suite of specialized tools for text analysis, image understanding, audio transcription, web search, Wikipedia extraction, and sandboxed code execution.  
Developed as the **final project for the Hugging Face Agents Course**, AtlasMind demonstrates a full agentic workflow built around classification, planning, tool routing, and structured synthesis.

AtlasMind uses three coordinated LLMs:
- **Gemini 2.5 Flash** for multimodal tasks (text + image reasoning)
- **GPT-4o-mini** (GitHub Models) for planning and step decomposition
- **GPT-4o** (GitHub Models) for final synthesis and answer construction

This multi-LLM design enables precise planning, grounded tool execution, and high-quality final output.

---

## Project Overview

AtlasMind combines the **LlamaIndex agent workflow** with a **custom multi-tool execution engine** to create a flexible and extensible reasoning system.  
A user query flows through semantic classification, planning, tool invocation, and final synthesis. Each stage is validated through Pydantic schemas, ensuring high reliability and predictable outputs.

This architecture allows AtlasMind to handle real-world tasks that span text, images, audio, external knowledge retrieval, and executable code. The agent can break down problems, use the correct tools, merge diverse outputs, and generate clean, validated responses.

### Key Capabilities

AtlasMind integrates:

- **Gemini 2.5 Flash** for multimodal reasoning (text + images)
- **GPT-4o-mini** for planning, decomposition, and follow-up generation
- **GPT-4o** for final structured synthesis
- **Semantic classifier** to determine reasoning mode and route tasks
- **SerpAPI Web Search** for external factual lookup
- **Wikipedia Retrieval Tool** for context-grounded knowledge extraction
- **Deepgram (nova-3)** for audio transcription
- **YouTubeTranscriptApi** for video transcript extraction
- **Judge0 sandbox** for safe code execution
- **Custom Pydantic models** for strict JSON output validation
- **FileFetcher** for downloading remote assets (audio, code, or images)

AtlasMind demonstrates how multiple modalities, tools, and models can be orchestrated into a unified agent capable of transparent, controllable reasoning.

---

## Tech Stack  

| Layer | Technology | Purpose |
|--------|------------|----------|
| **Agent Framework** | **LlamaIndex Agent Workflow** | Orchestrates classification, planning, tool routing, and synthesis |
| **Multimodal LLM** | **Gemini 2.5 Flash** | Image reasoning, multimodal understanding, semantic tasks |
| **Planning LLM** | **GPT-4o-mini (GitHub Models)** | Plan generation, step decomposition, and tool selection guidance |
| **Synthesis LLM** | **GPT-4o (GitHub Models)** | Final structured synthesis and merging of tool outputs |
| **Schema Validation** | **Pydantic v2** | Enforces strict JSON response formats and validation |
| **Semantic Classification** | Gemini-based classifier | Determines reasoning mode and workflow routing |
| **Web Search** | SerpAPI | External factual retrieval |
| **Wikipedia Retrieval** | Custom WikipediaTool | Extracts summaries and knowledge-grounded context |
| **Audio Transcription** | Deepgram (nova-3) | Converts audio input into structured text |
| **Video Transcription** | YouTubeTranscriptApi | Extracts transcripts from YouTube videos |
| **Code Execution** | Judge0 REST API | Sandboxed, language-agnostic code execution |
| **File Input** | Direct CLI file paths | Users provide images or files via `--file_path`, no network fetching |
| **Config Management** | python-dotenv | Loads API keys and environment variables |
| **Logging** | colorlog + custom logger utils | Rich, structured logs across tools and pipeline steps |

## Quick Start / Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/atlasmind.git
cd atlasmind
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# or
venv\Scripts\activate             # Windows
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
AtlasMind loads its API keys using `python-dotenv`.

Copy the example environment file:

#### macOS / Linux
```bash
cp example.env .env
```

#### Windows (Command Prompt)
```cmd
copy .env.example .env
```

#### Windows (PowerShell)
```powershell
Copy-Item .env.example .env
```

Fill in the required keys:

```env
GEMINI_API_KEY=
GITHUB_TOKEN=
SERPAPI_API_KEY=
DEEPGRAM_API_KEY=
```

## Quick Start / Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/atlasmind.git
cd atlasmind
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# or
venv\Scripts\activate           # Windows
```

### 3. Install AtlasMind in Editable Mode
```bash
pip install -e .
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy example file:

macOS / Linux:
```bash
cp .env.example .env
```

Windows CMD:
```cmd
copy .env.example .env
```

Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

Fill in:
```env
GEMINI_API_KEY=
GITHUB_TOKEN=
SERPAPI_API_KEY=
DEEPGRAM_API_KEY=
```

---

## Usage

AtlasMind's CLI supports only the following arguments:

- --question
- --file_path

### Text-only question
```bash
atlasmind --question "Explain how rainbows form"
```

### Image + question
```bash
atlasmind --question "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation." \
          --file_path ./tmp/sample/chess_position.png
```

### Audio transcription + reasoning
```bash
atlasmind --question "Hi, I was out sick from my classes on Friday, so I'm trying to figure out what I need to study for my Calculus mid-term next week. My friend from class sent me an audio recording of Professor Willowbrook giving out the recommended reading for the test, but my headphones are broken :(\n\nCould you please listen to the recording for me and tell me the page numbers I'm supposed to go over? I've attached a file called Homework.mp3 that has the recording. Please provide just the page numbers as a comma-delimited list. And please provide the list in ascending order." \
          --file_path ./tmp/sample/homework.mp3
```

### Code execution (via file)
```bash
atlasmind --question "Run this code" \
          --file_path tmp/sample/code.py
```

---

## Features
- Multimodal reasoning  
- Structured JSON output  
- LlamaIndex workflow orchestration  
- Planning + synthesis separation  
- Robust error handling  
- Extensible tool system  

---

## Testing
```bash
pytest
```

Validates:
- Tool reliability  
- Planning and classification  
- Audio/video transcription  
- Code execution  
- Output model validation  