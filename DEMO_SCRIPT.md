# 🎬 Data-to-Insight AI Analyst - Demo Video Script

## Video Duration: 2-3 Minutes

---

## 🎯 OPENING (0:00 - 0:20)

**[Show: Title slide or app landing page]**

> "Hello! I'm going to walk you through my Data-to-Insight AI Analyst - an intelligent prototype that automatically reads structured datasets and generates actionable business insights."

> "This solution was built to help mid-size retail clients who currently review weekly sales data manually. Instead of spending hours on spreadsheets, this AI agent automates trend detection, anomaly identification, and generates executive summaries with recommended actions."

---

## 📊 DEMO FLOW (0:20 - 2:20)

### Step 1: Upload / Input Data (0:20 - 0:40)
**[Show: Streamlit UI with file upload or URL input]**

> "The interface is simple - you can either provide a dataset URL or upload a CSV/Excel file directly. I'll use a sample retail sales dataset."

**Action:** Paste URL or upload `pokemon.csv` / sample sales CSV

> "The agent supports multiple analysis modes:
> - **Two-Phase Analysis** - comprehensive basic + deep pattern detection
> - **Basic Analysis** - trends, week-over-week changes, anomalies
> - **Deep Pattern Analysis** - segment-level cross-dimensional insights"

### Step 2: Agent Execution (0:40 - 1:00)
**[Show: Agent running with spinner/progress]**

> "When I click 'Run Agent', several things happen:
> 1. The data is securely loaded into an E2B cloud sandbox
> 2. The AI agent auto-discovers the schema - finding date columns, metrics, and groupings
> 3. It performs statistical analysis: WoW changes, trend detection, anomaly scoring
> 4. It generates visualizations and writes insights to JSON"

**Action:** Click "Run Agent in E2B Sandbox"

### Step 3: Results - Executive Summary (1:00 - 1:20)
**[Show: Business Summary section]**

> "First, we get an **Executive Business Summary** - a cohesive narrative summarizing the data story. For example:
> 'Overall revenue shows healthy growth of 3% with $4.1M total. While most regions perform well, Region B requires immediate attention due to declining performance.'"

### Step 4: Key Insights & Metrics (1:20 - 1:40)
**[Show: Overview metrics cards and insights]**

> "Next, the **Data Overview** shows key metrics at a glance:
> - Total revenue
> - Latest week-over-week change
> - Trend direction (increasing/decreasing)
> 
> The **Key Insights** section shows 3-5 data-driven findings with specific numbers."

### Step 5: Visualizations with AI Analysis (1:40 - 2:00)
**[Show: Charts with Vision LLM analysis]**

> "Here's a unique feature - **Vision LLM Analysis**. The AI doesn't just create charts, it analyzes them using GPT-4 Vision to explain:
> - Trend patterns and seasonality
> - Anomalies with business context  
> - Segment comparisons and performance gaps"

**[Scroll to show charts and AI explanations]**

### Step 6: Recommended Next Actions (2:00 - 2:15)
**[Show: Prioritized actions table]**

> "Finally, the **Recommended Next Actions** - prioritized, specific recommendations:
> - 🔴 URGENT: Investigate Region B's 28% decline
> - 🟡 HIGH: Replicate Region A's campaign in underperforming regions
> Each action includes expected impact and responsible owner."

### Step 7: Chat with Your Data (Optional) (2:15 - 2:30)
**[Show: Chat interface]**

> "Users can also **Chat with their Data** - asking follow-up questions like:
> 'Which region performed best last quarter?'
> 'Show me a pie chart of revenue by category'"

---

## 🎬 CLOSING (2:30 - 3:00)

**[Show: Architecture diagram or code structure]**

> "To summarize the architecture:
> - **Frontend**: Streamlit for interactive UI
> - **Backend**: Python with Pandas, OpenAI API
> - **Sandbox**: E2B for secure code execution
> - **AI Models**: GPT-4 for insights, GPT-4 Vision for chart analysis"

> "Key highlights:
> ✅ Fully automated insight generation
> ✅ Multiple analysis modes  
> ✅ Vision LLM for chart explanation
> ✅ Interactive chat capability
> ✅ Results caching for faster reruns"

> "Thank you for watching! The code is available on GitHub, and the app is deployed on Streamlit Cloud."

---

## 📋 DEMO CHECKLIST

Before recording:
- [ ] Test with sample dataset
- [ ] Ensure API keys are set
- [ ] Clear previous results for fresh demo
- [ ] Have sample questions ready for chat

Key screens to capture:
1. Landing page with input options
2. Analysis type selector
3. Running agent spinner
4. Executive Summary
5. Metrics overview cards
6. Time series chart + AI explanation
7. Group bar chart + AI explanation
8. Recommended Next Actions table
9. Chat interface (if time permits)
10. Architecture diagram

---

## 🗣️ SPEAKING TIPS

- Speak clearly and at moderate pace
- Pause on key insights
- Highlight numbers and specific values
- Show mouse pointer on important elements
- Keep energy positive and confident

