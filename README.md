# 📈 Data-to-Insight AI Analyst

An intelligent AI-powered data analysis agent that automatically reads structured datasets (CSV/Excel), identifies key trends and anomalies, and generates actionable business insights with visual explanations.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)
![E2B](https://img.shields.io/badge/E2B-Sandbox-orange.svg)

## ✨ Features

### Core Analysis
- **📊 Automatic Data Analysis**: Upload CSV/Excel files and get instant insights
- **🔍 Smart Schema Detection**: Auto-detects date columns, numeric metrics, and categorical dimensions
- **📈 Trend Analysis**: Weekly aggregation, WoW changes, trend direction via linear regression
- **⚠️ Anomaly Detection**: Z-score and IsolationForest methods for outlier detection
- **🎯 Actionable Recommendations**: Business-oriented suggestions based on data patterns

### AI-Powered Features
- **🤖 Remote Sandbox Execution**: Runs analysis securely in E2B cloud sandbox
- **👁️ Vision LLM Analysis**: GPT-4 Vision analyzes generated charts and provides detailed explanations
- **💡 Deep Pattern Detection**: Identifies segment-level anomalies and cross-dimensional patterns
- **📝 Natural Language Insights**: Plain English explanations of complex data patterns

### Business Intelligence Output
- **📋 Executive Business Summary**: Cohesive 2-3 sentence narrative summarizing the overall data story
- **🎯 Prioritized Next Actions**: 2-3 actionable recommendations with:
  - Priority level (Urgent/High/Medium)
  - Specific action to take
  - Expected business impact
  - Responsible owner/team

### Multiple Analysis Modes
| Mode | Description |
|------|-------------|
| 🔄 **Two-Phase Analysis** | Basic analysis + deep pattern detection (recommended) |
| 📊 **Basic Data Analysis** | Trends, WoW changes, anomaly detection |
| 🔬 **Deep Pattern Analysis** | Segment-level, cross-dimensional insights |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key (for AI features)
- E2B API key (for sandbox execution)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/dv0331/Ai-Data-Analyst.git
cd Ai-Data-Analyst
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your-openai-key" > .env
echo "E2B_API_KEY=your-e2b-key" >> .env
```

5. **Run the app**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
Data Analyst Agent/
├── app.py                 # Main Streamlit application
├── prompts.json           # AI prompts configuration (3 modes)
├── prompts.txt            # Human-readable prompt documentation
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore            # Git ignore rules
└── lib/
    ├── coding_agent.py   # Core agent logic
    ├── tools.py          # Tool implementations
    ├── tools_schemas.py  # Tool schemas for function calling
    └── ...
```

## 🎛️ Configuration

### Prompts
Edit `prompts.json` to customize agent behavior:

```json
{
  "data_analysis": { ... },    // Basic analysis prompt
  "deep_insights": { ... },    // Deep pattern detection prompt  
  "combined": { ... }          // Two-phase analysis (default)
}
```

### Analysis Settings
Adjust in the Streamlit sidebar:
- Time grain (Daily/Weekly/Monthly)
- Anomaly detection method (Z-score/IsolationForest)
- Z-score threshold
- IsolationForest contamination rate

## 📊 Sample Output

### Executive Business Summary
```
Overall business performance is strong with $4.1M total revenue and a healthy +3% WoW 
growth trend. Region A leads performance while Region B shows concerning -28% decline 
requiring immediate attention. The seasonal uptick in Q4 presents opportunity for 
targeted promotional activities.
```

### Key Insights Generated
```
✅ Total revenue: $4,103,231
✅ Latest WoW change: +3.01%
✅ Trend direction: Increasing (slope: +6,289/week)
✅ Anomaly detected: Week of Dec 15 (Region C -28%)
```

### Recommended Next Actions
The system generates prioritized, actionable recommendations:

| Priority | Action | Expected Impact | Owner |
|----------|--------|-----------------|-------|
| 🔴 URGENT | Investigate Region B's 28% revenue decline | Potential $50K recovery | Sales Team |
| 🟡 HIGH | Replicate Region A's campaign in underperforming regions | 15% lift in target regions | Marketing |
| 🟢 MEDIUM | Develop Q1 promotional calendar based on patterns | Better resource allocation | Planning |

### AI Chart Analysis
The Vision LLM provides detailed analysis of each chart:
- **Trend patterns** and seasonal cycles
- **Anomaly explanations** with business context
- **Segment comparisons** and performance gaps
- **Actionable recommendations** based on visual patterns

## 🔧 API Keys Required

| Service | Purpose | Get Key |
|---------|---------|---------|
| OpenAI | GPT-4 for insights + Vision for chart analysis | [platform.openai.com](https://platform.openai.com) |
| E2B | Secure sandbox for code execution | [e2b.dev](https://e2b.dev) |

## 🏗️ Architecture

![Architecture Diagram](Architecture.svg)

<details>
<summary>View Text-based Architecture</summary>

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│  E2B        │────▶│   OpenAI    │
│   Dataset   │     │  Sandbox    │     │   GPT-4     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │   Charts    │────▶│   Vision    │
                    │   (PNG)     │     │   Analysis  │
                    └─────────────┘     └─────────────┘
                           │                   │
                           └───────────────────┘
                                    │
                                    ▼
                           ┌─────────────┐
                           │  Streamlit  │
                           │     UI      │
                           └─────────────┘
```
</details>

## 📝 Design Approach

1. **Ingest**: Load CSV/Excel file
2. **Infer Schema**: Detect date/metric/group columns
3. **Aggregate**: Weekly time series aggregation
4. **Analyze**: Trends, WoW changes, anomalies
5. **Detect Patterns**: Segment-level deep analysis
6. **Generate Insights**: Plain English findings
7. **Visualize**: Charts with AI explanations
8. **Recommend**: Actionable next steps

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - feel free to use this for your own projects.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI GPT-4](https://openai.com/)
- Sandboxed by [E2B](https://e2b.dev/)
