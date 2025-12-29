# 📚 Data-to-Insight AI Analyst - Project Overview

## Reading Matter for Demo Preparation

---

## 🎯 Problem Statement

**Client Context:** Mid-size retail client that currently reviews weekly sales data manually to detect trends and performance issues.

**Challenge:** Manual analysis is time-consuming, error-prone, and doesn't scale. The client wants automation that can:
- Read uploaded CSV/Excel data
- Identify trends (e.g., "Region B's revenue decreased by 12% WoW")
- Generate concise business summaries with recommended actions

**Solution:** An AI-powered "Data Analyst" agent that automates the entire process.

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB APPLICATION                      │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐     │
│  │ File Upload │───▶│   E2B Cloud  │───▶│    OpenAI GPT    │     │
│  │  or URL     │    │   Sandbox    │    │  (Code + Vision) │     │
│  └─────────────┘    └──────────────┘    └──────────────────┘     │
│         │                  │                     │                │
│         ▼                  ▼                     ▼                │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐     │
│  │  Schema     │    │  Analysis    │    │  Chart Analysis  │     │
│  │  Detection  │    │  (Pandas)    │    │  (GPT-4 Vision)  │     │
│  └─────────────┘    └──────────────┘    └──────────────────┘     │
│                           │                     │                 │
│                           ▼                     ▼                 │
│                    ┌──────────────────────────────────┐          │
│                    │        INSIGHTS OUTPUT           │          │
│                    │  • Executive Summary             │          │
│                    │  • Metrics Dashboard             │          │
│                    │  • Charts + AI Explanations      │          │
│                    │  • Recommended Actions           │          │
│                    │  • Interactive Chat              │          │
│                    └──────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Features

### 1. Core Functionality (Required)
| Feature | Description |
|---------|-------------|
| **File Upload** | CSV or Excel file support |
| **Auto Schema Detection** | Identifies date columns, numeric metrics, categorical dimensions |
| **Trend Analysis** | Weekly aggregation, WoW % change, linear regression slope |
| **Insight Generation** | 3-5 business insights with specific numbers |
| **Recommendations** | 1-2 actionable business recommendations |

### 2. Optional Enhancements (Implemented)
| Feature | Description |
|---------|-------------|
| **Anomaly Detection** | Z-score and IsolationForest methods |
| **Vision LLM Analysis** | GPT-4 Vision analyzes charts and explains patterns |
| **Chat Interface** | Natural language queries ("Which region performed best?") |
| **Results Caching** | Cache results per analysis type for faster reruns |
| **Multiple Analysis Modes** | Basic, Deep Pattern, Two-Phase analysis options |

---

## 📊 Analysis Modes

### Mode 1: Basic Data Analysis
- Trends, WoW changes, anomaly detection
- Time series chart + group bar chart
- 3-5 insights + 1-2 recommendations

### Mode 2: Deep Pattern Analysis
- Segment-level anomaly detection
- Cross-dimensional patterns
- Heatmap visualization
- Detailed pattern explanations

### Mode 3: Two-Phase Analysis (Recommended)
- Combines Basic + Deep Pattern
- Most comprehensive output
- Best for executive presentations

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit (Python web framework) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **AI/LLM** | OpenAI GPT-4o (insights), GPT-4 Vision (charts) |
| **Sandbox** | E2B Cloud (secure code execution) |
| **Anomaly Detection** | scikit-learn (IsolationForest), scipy (Z-score) |
| **Deployment** | Streamlit Cloud |

---

## 📁 Code Structure

```
Data Analyst Agent/
├── app.py                 # Main Streamlit application (2000+ lines)
│   ├── Data loading & schema inference
│   ├── Local analysis functions
│   ├── Remote sandbox execution
│   ├── Vision LLM chart analysis
│   ├── Chat interface
│   └── Results caching
├── prompts.json           # AI prompts for 3 analysis modes
├── lib/
│   ├── coding_agent.py    # Agent loop with function calling
│   ├── tools.py           # Tool implementations (execute_code, etc.)
│   ├── tools_schemas.py   # OpenAI function schemas
│   └── prompts.py         # Prompt templates
├── .streamlit/
│   ├── config.toml        # Theme & server settings
│   └── secrets.toml.example # API keys template
└── Samples/               # Sample PDF outputs
```

---

## 🔄 Data Flow

```
1. INPUT
   User uploads CSV/Excel or provides URL
   
2. SCHEMA INFERENCE
   - Detect date columns (contains 'date', 'week', 'time')
   - Find numeric metrics (pick largest sum - likely revenue)
   - Identify categorical dimensions (region, category, channel)

3. AGENT EXECUTION (in E2B Sandbox)
   - Load data with Pandas
   - Aggregate by week/time period
   - Calculate: total, WoW change, trend slope
   - Detect anomalies (Z-score >= 2.5)
   - Generate charts (matplotlib → PNG)
   - Write insights.json

4. VISION ANALYSIS
   - GPT-4 Vision reads chart images
   - Provides natural language explanations
   - Identifies patterns, anomalies, recommendations

5. OUTPUT
   - Executive Business Summary
   - Metrics Dashboard
   - Charts with AI explanations
   - Prioritized Next Actions
   - Optional: Chat interface
```

---

## 💡 Key Design Decisions

### 1. Why E2B Sandbox?
- **Security**: Code executes in isolated cloud environment
- **Flexibility**: Agent can install packages, run arbitrary Python
- **No local dependencies**: Works on any deployment platform

### 2. Why GPT-4 Vision for Charts?
- Provides **human-like explanations** of visual patterns
- Catches nuances that code-based analysis might miss
- Adds value beyond just showing charts

### 3. Why Multiple Analysis Modes?
- Different use cases need different depths
- Basic: Quick overview (faster, cheaper)
- Deep: Thorough investigation (more tokens)
- Two-Phase: Best of both worlds

### 4. Why Results Caching?
- Analysis takes 30-60 seconds
- Caching allows switching between modes without re-running
- Clears automatically when dataset changes

---

## 📈 Sample Output

### Executive Summary Example
> "Overall business performance is strong with $4.1M total revenue and a healthy +3% WoW growth trend. Region A leads performance while Region B shows concerning -28% decline requiring immediate attention. The seasonal uptick in Q4 presents opportunity for targeted promotional activities."

### Key Insights Example
1. Total revenue: $4,103,231
2. Latest WoW change: +3.01%
3. Trend direction: Increasing (slope: +6,289/week)
4. Anomaly detected: Week of Dec 15 (Region C -28%)
5. Top performer: Region A with $1.2M revenue

### Next Actions Example
| Priority | Action | Impact | Owner |
|----------|--------|--------|-------|
| 🔴 URGENT | Investigate Region B's 28% decline | $50K recovery | Sales Team |
| 🟡 HIGH | Replicate Region A campaign | 15% lift | Marketing |
| 🟢 MEDIUM | Develop Q1 promotional calendar | Better planning | Planning |

---

## ⚠️ Challenges & Learnings

### Challenge 1: Binary File Handling
**Problem:** Reading PNG files from sandbox returned corrupted data
**Solution:** Base64 encoding/decoding with multiple fallback strategies

### Challenge 2: LLM Output Consistency
**Problem:** GPT-4 sometimes returns malformed JSON or exceeds pattern limits
**Solution:** Detailed prompts with EXACT structure, MAX limits in instructions

### Challenge 3: Schema Inference
**Problem:** Messy datasets with duplicate columns, mixed types
**Solution:** `sanitize_dataframe()` function, robust column detection

### Challenge 4: State Management
**Problem:** Streamlit reruns entire app on interaction
**Solution:** `st.session_state` for caching results per analysis type

---

## 🚀 Future Enhancements (If More Time)

1. **Multi-file comparison** - Compare this week vs last week datasets
2. **Scheduled reports** - Automated weekly email summaries
3. **Custom prompts** - User can edit analysis prompts
4. **Export to PDF** - One-click executive report generation
5. **Database integration** - Direct SQL query support
6. **Alerting** - Slack/Teams notifications for anomalies

---

## 📊 Evaluation Focus Areas

| Focus Area | Weight | How We Address It |
|------------|--------|-------------------|
| Integration & Execution | 40% | Full end-to-end pipeline, Vision LLM, sandbox execution |
| Clarity of Insights | 30% | Executive summary, specific numbers, prioritized actions |
| Communication | 15% | Clean UI, clear explanations, demo video |
| Creativity & Initiative | 15% | Vision analysis, chat interface, results caching |

---

## 🔗 Links

- **GitHub**: https://github.com/dv0331/Ai-Data-Analyst
- **Live Demo**: [Streamlit Cloud URL]
- **Sample Outputs**: See `/Samples/` folder for PDF examples

