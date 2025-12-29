' ============================================================================
' VBA CODE TO CREATE 5-SLIDE PRESENTATION FOR DATA-TO-INSIGHT AI ANALYST
' ============================================================================
' 
' HOW TO USE:
' 1. Open PowerPoint
' 2. Press Alt + F11 to open VBA Editor
' 3. Insert > Module
' 4. Paste this code
' 5. Press F5 or Run > Run Sub/UserForm
' 6. Select "CreateDataAnalystPresentation" and click Run
'
' The presentation will be created with all 5 slides pre-formatted.
' You can then customize colors, add screenshots, and fine-tune content.
' ============================================================================

Sub CreateDataAnalystPresentation()
    
    Dim pptApp As Object
    Dim pptPres As Object
    Dim sld As Object
    Dim shp As Object
    Dim tbl As Object
    
    ' Create PowerPoint application
    On Error Resume Next
    Set pptApp = GetObject(, "PowerPoint.Application")
    If pptApp Is Nothing Then
        Set pptApp = CreateObject("PowerPoint.Application")
    End If
    On Error GoTo 0
    
    pptApp.Visible = True
    
    ' Create new presentation
    Set pptPres = pptApp.Presentations.Add
    
    ' Set slide dimensions (16:9 widescreen)
    pptPres.PageSetup.SlideWidth = 960
    pptPres.PageSetup.SlideHeight = 540
    
    ' ========================================
    ' SLIDE 1: Problem Understanding & Objective
    ' ========================================
    Set sld = pptPres.Slides.Add(1, 12) ' ppLayoutBlank
    
    ' Title
    Set shp = sld.Shapes.AddTextbox(1, 40, 30, 880, 60)
    With shp.TextFrame.TextRange
        .Text = "Problem Understanding & Objective"
        .Font.Size = 36
        .Font.Bold = True
        .Font.Color.RGB = RGB(32, 56, 100)
    End With
    
    ' Subtitle
    Set shp = sld.Shapes.AddTextbox(1, 40, 95, 880, 30)
    With shp.TextFrame.TextRange
        .Text = "Data-to-Insight AI Analyst | Automated Business Intelligence"
        .Font.Size = 16
        .Font.Color.RGB = RGB(100, 100, 100)
    End With
    
    ' Content box - Problem
    Set shp = sld.Shapes.AddShape(5, 40, 140, 420, 180) ' msoShapeRoundedRectangle
    shp.Fill.ForeColor.RGB = RGB(240, 248, 255)
    shp.Line.ForeColor.RGB = RGB(32, 56, 100)
    
    Set shp = sld.Shapes.AddTextbox(1, 55, 150, 390, 160)
    With shp.TextFrame.TextRange
        .Text = "THE PROBLEM" & vbCrLf & vbCrLf & _
                Chr(149) & " Mid-size retail client reviews weekly sales data manually" & vbCrLf & _
                Chr(149) & " Hours spent detecting trends and performance issues" & vbCrLf & _
                Chr(149) & " Manual process is slow, error-prone, and doesn't scale" & vbCrLf & _
                Chr(149) & " No automated anomaly detection or recommendations"
        .Font.Size = 14
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 18
        .Paragraphs(1).Font.Color.RGB = RGB(192, 0, 0)
    End With
    
    ' Content box - Solution
    Set shp = sld.Shapes.AddShape(5, 500, 140, 420, 180) ' msoShapeRoundedRectangle
    shp.Fill.ForeColor.RGB = RGB(240, 255, 240)
    shp.Line.ForeColor.RGB = RGB(0, 128, 0)
    
    Set shp = sld.Shapes.AddTextbox(1, 515, 150, 390, 160)
    With shp.TextFrame.TextRange
        .Text = "THE SOLUTION" & vbCrLf & vbCrLf & _
                Chr(149) & " AI-powered 'Data Analyst' agent" & vbCrLf & _
                Chr(149) & " Upload CSV/Excel " & Chr(8594) & " Get instant insights" & vbCrLf & _
                Chr(149) & " Automatic trend detection (WoW changes)" & vbCrLf & _
                Chr(149) & " Anomaly identification with business context" & vbCrLf & _
                Chr(149) & " Executive summary + prioritized actions"
        .Font.Size = 14
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 18
        .Paragraphs(1).Font.Color.RGB = RGB(0, 128, 0)
    End With
    
    ' Objective
    Set shp = sld.Shapes.AddShape(5, 40, 340, 880, 80) ' msoShapeRoundedRectangle
    shp.Fill.ForeColor.RGB = RGB(32, 56, 100)
    
    Set shp = sld.Shapes.AddTextbox(1, 55, 355, 850, 60)
    With shp.TextFrame.TextRange
        .Text = "OBJECTIVE: Design a proof-of-concept AI Data Analyst that reads uploaded data, " & _
                "identifies trends & anomalies, and generates a concise business summary with 2-3 recommended actions."
        .Font.Size = 14
        .Font.Color.RGB = RGB(255, 255, 255)
        .Font.Bold = True
    End With
    
    ' Footer
    Set shp = sld.Shapes.AddTextbox(1, 40, 450, 200, 25)
    With shp.TextFrame.TextRange
        .Text = "Tech Stack: Python | Streamlit | OpenAI | E2B"
        .Font.Size = 10
        .Font.Color.RGB = RGB(128, 128, 128)
    End With
    
    ' ========================================
    ' SLIDE 2: Solution Architecture & Design Flow
    ' ========================================
    Set sld = pptPres.Slides.Add(2, 12) ' ppLayoutBlank
    
    ' Title
    Set shp = sld.Shapes.AddTextbox(1, 40, 30, 880, 60)
    With shp.TextFrame.TextRange
        .Text = "Solution Architecture & Design Flow"
        .Font.Size = 36
        .Font.Bold = True
        .Font.Color.RGB = RGB(32, 56, 100)
    End With
    
    ' Architecture boxes - Row 1
    ' Input box
    Set shp = sld.Shapes.AddShape(5, 40, 110, 140, 70)
    shp.Fill.ForeColor.RGB = RGB(52, 152, 219)
    Set shp = sld.Shapes.AddTextbox(1, 45, 125, 130, 50)
    With shp.TextFrame.TextRange
        .Text = "1. FILE UPLOAD" & vbCrLf & "CSV / Excel / URL"
        .Font.Size = 11
        .Font.Color.RGB = RGB(255, 255, 255)
        .Font.Bold = True
        .ParagraphFormat.Alignment = 2 ' Center
    End With
    
    ' Arrow
    Set shp = sld.Shapes.AddShape(36, 185, 130, 40, 30) ' msoShapeRightArrow
    shp.Fill.ForeColor.RGB = RGB(100, 100, 100)
    
    ' Schema Detection box
    Set shp = sld.Shapes.AddShape(5, 230, 110, 140, 70)
    shp.Fill.ForeColor.RGB = RGB(155, 89, 182)
    Set shp = sld.Shapes.AddTextbox(1, 235, 125, 130, 50)
    With shp.TextFrame.TextRange
        .Text = "2. SCHEMA DETECT" & vbCrLf & "Date/Metric/Groups"
        .Font.Size = 11
        .Font.Color.RGB = RGB(255, 255, 255)
        .Font.Bold = True
        .ParagraphFormat.Alignment = 2
    End With
    
    ' Arrow
    Set shp = sld.Shapes.AddShape(36, 375, 130, 40, 30)
    shp.Fill.ForeColor.RGB = RGB(100, 100, 100)
    
    ' E2B Sandbox box
    Set shp = sld.Shapes.AddShape(5, 420, 110, 140, 70)
    shp.Fill.ForeColor.RGB = RGB(230, 126, 34)
    Set shp = sld.Shapes.AddTextbox(1, 425, 125, 130, 50)
    With shp.TextFrame.TextRange
        .Text = "3. E2B SANDBOX" & vbCrLf & "Secure Execution"
        .Font.Size = 11
        .Font.Color.RGB = RGB(255, 255, 255)
        .Font.Bold = True
        .ParagraphFormat.Alignment = 2
    End With
    
    ' Arrow
    Set shp = sld.Shapes.AddShape(36, 565, 130, 40, 30)
    shp.Fill.ForeColor.RGB = RGB(100, 100, 100)
    
    ' Analysis box
    Set shp = sld.Shapes.AddShape(5, 610, 110, 140, 70)
    shp.Fill.ForeColor.RGB = RGB(46, 204, 113)
    Set shp = sld.Shapes.AddTextbox(1, 615, 125, 130, 50)
    With shp.TextFrame.TextRange
        .Text = "4. ANALYSIS" & vbCrLf & "Pandas + Stats"
        .Font.Size = 11
        .Font.Color.RGB = RGB(255, 255, 255)
        .Font.Bold = True
        .ParagraphFormat.Alignment = 2
    End With
    
    ' Arrow
    Set shp = sld.Shapes.AddShape(36, 755, 130, 40, 30)
    shp.Fill.ForeColor.RGB = RGB(100, 100, 100)
    
    ' OpenAI box
    Set shp = sld.Shapes.AddShape(5, 800, 110, 120, 70)
    shp.Fill.ForeColor.RGB = RGB(0, 166, 125)
    Set shp = sld.Shapes.AddTextbox(1, 805, 125, 110, 50)
    With shp.TextFrame.TextRange
        .Text = "5. OPENAI" & vbCrLf & "GPT-4 + Vision"
        .Font.Size = 11
        .Font.Color.RGB = RGB(255, 255, 255)
        .Font.Bold = True
        .ParagraphFormat.Alignment = 2
    End With
    
    ' Row 2 - Output components
    Set shp = sld.Shapes.AddTextbox(1, 40, 200, 880, 25)
    With shp.TextFrame.TextRange
        .Text = "OUTPUT COMPONENTS"
        .Font.Size = 14
        .Font.Bold = True
        .Font.Color.RGB = RGB(32, 56, 100)
    End With
    
    ' Output boxes
    Dim outputLabels As Variant
    outputLabels = Array("Executive Summary", "Metrics Dashboard", "Trend Charts", "AI Chart Analysis", "Next Actions", "Chat Interface")
    Dim i As Integer
    For i = 0 To 5
        Set shp = sld.Shapes.AddShape(5, 40 + (i * 150), 230, 135, 50)
        shp.Fill.ForeColor.RGB = RGB(240, 240, 240)
        shp.Line.ForeColor.RGB = RGB(32, 56, 100)
        Set shp = sld.Shapes.AddTextbox(1, 45 + (i * 150), 240, 125, 35)
        With shp.TextFrame.TextRange
            .Text = outputLabels(i)
            .Font.Size = 11
            .Font.Bold = True
            .ParagraphFormat.Alignment = 2
        End With
    Next i
    
    ' Data flow description
    Set shp = sld.Shapes.AddShape(5, 40, 300, 880, 120)
    shp.Fill.ForeColor.RGB = RGB(250, 250, 250)
    shp.Line.ForeColor.RGB = RGB(200, 200, 200)
    
    Set shp = sld.Shapes.AddTextbox(1, 55, 310, 850, 100)
    With shp.TextFrame.TextRange
        .Text = "DATA FLOW:" & vbCrLf & vbCrLf & _
                "1. User uploads file or provides URL " & Chr(8594) & " Data sent to E2B Cloud Sandbox (never to LLM directly)" & vbCrLf & _
                "2. AI Agent auto-discovers schema: date columns, numeric metrics (picks largest sum), categorical dimensions" & vbCrLf & _
                "3. Analysis: Weekly aggregation " & Chr(8594) & " WoW changes " & Chr(8594) & " Trend slope (linear regression) " & Chr(8594) & " Anomaly detection (Z-score " & Chr(8805) & " 2.5)" & vbCrLf & _
                "4. Charts generated (matplotlib) " & Chr(8594) & " GPT-4 Vision analyzes images " & Chr(8594) & " Natural language explanations" & vbCrLf & _
                "5. insights.json output with business_summary, insights, recommendations, next_actions"
        .Font.Size = 11
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 12
        .Paragraphs(1).Font.Color.RGB = RGB(32, 56, 100)
    End With
    
    ' Note: Add Architecture.svg screenshot here
    Set shp = sld.Shapes.AddTextbox(1, 40, 430, 300, 25)
    With shp.TextFrame.TextRange
        .Text = "[Insert Architecture.svg diagram here]"
        .Font.Size = 10
        .Font.Italic = True
        .Font.Color.RGB = RGB(150, 150, 150)
    End With
    
    ' ========================================
    ' SLIDE 3: Implementation Highlights
    ' ========================================
    Set sld = pptPres.Slides.Add(3, 12)
    
    ' Title
    Set shp = sld.Shapes.AddTextbox(1, 40, 30, 880, 60)
    With shp.TextFrame.TextRange
        .Text = "Implementation Highlights"
        .Font.Size = 36
        .Font.Bold = True
        .Font.Color.RGB = RGB(32, 56, 100)
    End With
    
    ' Column 1 - Technical Decisions
    Set shp = sld.Shapes.AddShape(5, 40, 100, 290, 320)
    shp.Fill.ForeColor.RGB = RGB(240, 248, 255)
    shp.Line.ForeColor.RGB = RGB(52, 152, 219)
    
    Set shp = sld.Shapes.AddTextbox(1, 50, 110, 270, 300)
    With shp.TextFrame.TextRange
        .Text = Chr(128295) & " TECHNICAL STACK" & vbCrLf & vbCrLf & _
                Chr(149) & " Frontend: Streamlit" & vbCrLf & _
                Chr(149) & " Data: Pandas, NumPy" & vbCrLf & _
                Chr(149) & " Viz: Matplotlib, Seaborn" & vbCrLf & _
                Chr(149) & " AI: OpenAI GPT-4o" & vbCrLf & _
                Chr(149) & " Sandbox: E2B Cloud" & vbCrLf & _
                Chr(149) & " Anomaly: sklearn, scipy" & vbCrLf & vbCrLf & _
                Chr(128293) & " WHY E2B?" & vbCrLf & _
                "- Secure isolated execution" & vbCrLf & _
                "- Agent can pip install" & vbCrLf & _
                "- Data never sent to LLM"
        .Font.Size = 12
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 14
        .Paragraphs(9).Font.Bold = True
        .Paragraphs(9).Font.Size = 14
    End With
    
    ' Column 2 - AI Logic
    Set shp = sld.Shapes.AddShape(5, 345, 100, 290, 320)
    shp.Fill.ForeColor.RGB = RGB(240, 255, 240)
    shp.Line.ForeColor.RGB = RGB(46, 204, 113)
    
    Set shp = sld.Shapes.AddTextbox(1, 355, 110, 270, 300)
    With shp.TextFrame.TextRange
        .Text = Chr(129302) & " AI LOGIC" & vbCrLf & vbCrLf & _
                "3 Analysis Modes:" & vbCrLf & _
                Chr(9679) & " Basic: Trends, WoW, Anomalies" & vbCrLf & _
                Chr(9679) & " Deep: Segment patterns" & vbCrLf & _
                Chr(9679) & " Two-Phase: Combined" & vbCrLf & vbCrLf & _
                Chr(128065) & " VISION LLM" & vbCrLf & _
                "- GPT-4 Vision reads charts" & vbCrLf & _
                "- Explains patterns visually" & vbCrLf & _
                "- Catches nuances code misses" & vbCrLf & vbCrLf & _
                Chr(128172) & " CHAT INTERFACE" & vbCrLf & _
                "- Follow-up questions" & vbCrLf & _
                "- Dynamic visualizations"
        .Font.Size = 12
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 14
        .Paragraphs(8).Font.Bold = True
        .Paragraphs(12).Font.Bold = True
    End With
    
    ' Column 3 - Code Snippet
    Set shp = sld.Shapes.AddShape(5, 650, 100, 270, 320)
    shp.Fill.ForeColor.RGB = RGB(40, 44, 52)
    shp.Line.ForeColor.RGB = RGB(100, 100, 100)
    
    Set shp = sld.Shapes.AddTextbox(1, 660, 110, 250, 300)
    With shp.TextFrame.TextRange
        .Text = "# Key Code Pattern" & vbCrLf & vbCrLf & _
                "# Schema Detection" & vbCrLf & _
                "date_cols = [c for c in cols" & vbCrLf & _
                "  if 'date' in c.lower()]" & vbCrLf & vbCrLf & _
                "# Trend Analysis" & vbCrLf & _
                "slope, _ = linregress(" & vbCrLf & _
                "  range(len(ts)), ts)" & vbCrLf & vbCrLf & _
                "# Anomaly Detection" & vbCrLf & _
                "z_scores = zscore(values)" & vbCrLf & _
                "anomalies = |z| >= 2.5" & vbCrLf & vbCrLf & _
                "# Vision Analysis" & vbCrLf & _
                "vision_response = " & vbCrLf & _
                "  gpt4o.analyze(chart)"
        .Font.Size = 10
        .Font.Name = "Consolas"
        .Font.Color.RGB = RGB(171, 178, 191)
    End With
    
    ' Footer note
    Set shp = sld.Shapes.AddTextbox(1, 40, 440, 880, 30)
    With shp.TextFrame.TextRange
        .Text = Chr(128161) & " Key Highlight: Results are cached per analysis type - switching modes doesn't require re-running if data hasn't changed"
        .Font.Size = 12
        .Font.Color.RGB = RGB(32, 56, 100)
        .Font.Bold = True
    End With
    
    ' ========================================
    ' SLIDE 4: Challenges & Learnings
    ' ========================================
    Set sld = pptPres.Slides.Add(4, 12)
    
    ' Title
    Set shp = sld.Shapes.AddTextbox(1, 40, 30, 880, 60)
    With shp.TextFrame.TextRange
        .Text = "Challenges & Learnings"
        .Font.Size = 36
        .Font.Bold = True
        .Font.Color.RGB = RGB(32, 56, 100)
    End With
    
    ' Challenge 1
    Set shp = sld.Shapes.AddShape(5, 40, 100, 430, 100)
    shp.Fill.ForeColor.RGB = RGB(255, 245, 238)
    shp.Line.ForeColor.RGB = RGB(230, 126, 34)
    
    Set shp = sld.Shapes.AddTextbox(1, 50, 108, 410, 85)
    With shp.TextFrame.TextRange
        .Text = Chr(9888) & " CHALLENGE 1: Binary File Handling" & vbCrLf & _
                "Problem: PNG files from sandbox returned corrupted" & vbCrLf & _
                Chr(10004) & " Solution: Base64 encoding with multiple fallback strategies"
        .Font.Size = 12
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 14
        .Paragraphs(3).Font.Color.RGB = RGB(0, 128, 0)
    End With
    
    ' Challenge 2
    Set shp = sld.Shapes.AddShape(5, 490, 100, 430, 100)
    shp.Fill.ForeColor.RGB = RGB(255, 245, 238)
    shp.Line.ForeColor.RGB = RGB(230, 126, 34)
    
    Set shp = sld.Shapes.AddTextbox(1, 500, 108, 410, 85)
    With shp.TextFrame.TextRange
        .Text = Chr(9888) & " CHALLENGE 2: LLM Output Consistency" & vbCrLf & _
                "Problem: GPT-4 sometimes returns malformed JSON" & vbCrLf & _
                Chr(10004) & " Solution: Detailed prompts with EXACT structure, MAX limits"
        .Font.Size = 12
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 14
        .Paragraphs(3).Font.Color.RGB = RGB(0, 128, 0)
    End With
    
    ' Challenge 3
    Set shp = sld.Shapes.AddShape(5, 40, 215, 430, 100)
    shp.Fill.ForeColor.RGB = RGB(255, 245, 238)
    shp.Line.ForeColor.RGB = RGB(230, 126, 34)
    
    Set shp = sld.Shapes.AddTextbox(1, 50, 223, 410, 85)
    With shp.TextFrame.TextRange
        .Text = Chr(9888) & " CHALLENGE 3: Schema Inference" & vbCrLf & _
                "Problem: Messy datasets with duplicate cols, mixed types" & vbCrLf & _
                Chr(10004) & " Solution: sanitize_dataframe() + robust column detection"
        .Font.Size = 12
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 14
        .Paragraphs(3).Font.Color.RGB = RGB(0, 128, 0)
    End With
    
    ' Challenge 4
    Set shp = sld.Shapes.AddShape(5, 490, 215, 430, 100)
    shp.Fill.ForeColor.RGB = RGB(255, 245, 238)
    shp.Line.ForeColor.RGB = RGB(230, 126, 34)
    
    Set shp = sld.Shapes.AddTextbox(1, 500, 223, 410, 85)
    With shp.TextFrame.TextRange
        .Text = Chr(9888) & " CHALLENGE 4: State Management" & vbCrLf & _
                "Problem: Streamlit reruns app on every interaction" & vbCrLf & _
                Chr(10004) & " Solution: st.session_state caching per analysis type"
        .Font.Size = 12
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 14
        .Paragraphs(3).Font.Color.RGB = RGB(0, 128, 0)
    End With
    
    ' Learnings section
    Set shp = sld.Shapes.AddShape(5, 40, 340, 880, 110)
    shp.Fill.ForeColor.RGB = RGB(240, 248, 255)
    shp.Line.ForeColor.RGB = RGB(32, 56, 100)
    
    Set shp = sld.Shapes.AddTextbox(1, 55, 350, 850, 95)
    With shp.TextFrame.TextRange
        .Text = Chr(128218) & " KEY LEARNINGS & TAKEAWAYS" & vbCrLf & vbCrLf & _
                Chr(149) & " Sandbox execution adds security but requires careful binary handling (base64 encode/decode)" & vbCrLf & _
                Chr(149) & " Vision LLMs add significant value - they catch patterns that code analysis misses" & vbCrLf & _
                Chr(149) & " Prompt engineering is crucial - explicit structure, limits, and examples improve consistency" & vbCrLf & _
                Chr(149) & " Caching strategies save API costs and improve UX (30-60 sec analysis should persist)"
        .Font.Size = 12
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 14
        .Paragraphs(1).Font.Color.RGB = RGB(32, 56, 100)
    End With
    
    ' ========================================
    ' SLIDE 5: Demo Summary & Next Steps
    ' ========================================
    Set sld = pptPres.Slides.Add(5, 12)
    
    ' Title
    Set shp = sld.Shapes.AddTextbox(1, 40, 30, 880, 60)
    With shp.TextFrame.TextRange
        .Text = "Demo Summary & Next Steps"
        .Font.Size = 36
        .Font.Bold = True
        .Font.Color.RGB = RGB(32, 56, 100)
    End With
    
    ' What was delivered
    Set shp = sld.Shapes.AddShape(5, 40, 100, 430, 200)
    shp.Fill.ForeColor.RGB = RGB(240, 255, 240)
    shp.Line.ForeColor.RGB = RGB(46, 204, 113)
    
    Set shp = sld.Shapes.AddTextbox(1, 50, 108, 410, 185)
    With shp.TextFrame.TextRange
        .Text = Chr(10004) & " DELIVERED FEATURES" & vbCrLf & vbCrLf & _
                Chr(9679) & " CSV/Excel file upload or URL input" & vbCrLf & _
                Chr(9679) & " Auto schema detection" & vbCrLf & _
                Chr(9679) & " Trend analysis (WoW changes, slope)" & vbCrLf & _
                Chr(9679) & " Anomaly detection (Z-score, IsolationForest)" & vbCrLf & _
                Chr(9679) & " Vision LLM chart analysis" & vbCrLf & _
                Chr(9679) & " Executive summary + next actions" & vbCrLf & _
                Chr(9679) & " Interactive chat with data" & vbCrLf & _
                Chr(9679) & " 3 analysis modes + results caching"
        .Font.Size = 12
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 14
        .Paragraphs(1).Font.Color.RGB = RGB(0, 128, 0)
    End With
    
    ' Future enhancements
    Set shp = sld.Shapes.AddShape(5, 490, 100, 430, 200)
    shp.Fill.ForeColor.RGB = RGB(240, 248, 255)
    shp.Line.ForeColor.RGB = RGB(52, 152, 219)
    
    Set shp = sld.Shapes.AddTextbox(1, 500, 108, 410, 185)
    With shp.TextFrame.TextRange
        .Text = Chr(128300) & " FUTURE ENHANCEMENTS" & vbCrLf & vbCrLf & _
                Chr(9679) & " Multi-file comparison (week vs week)" & vbCrLf & _
                Chr(9679) & " Scheduled automated reports" & vbCrLf & _
                Chr(9679) & " Custom user-editable prompts" & vbCrLf & _
                Chr(9679) & " One-click PDF export" & vbCrLf & _
                Chr(9679) & " Direct SQL/database integration" & vbCrLf & _
                Chr(9679) & " Slack/Teams alerting for anomalies" & vbCrLf & _
                Chr(9679) & " Role-based access control"
        .Font.Size = 12
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 14
        .Paragraphs(1).Font.Color.RGB = RGB(52, 152, 219)
    End With
    
    ' Links section
    Set shp = sld.Shapes.AddShape(5, 40, 320, 880, 80)
    shp.Fill.ForeColor.RGB = RGB(32, 56, 100)
    
    Set shp = sld.Shapes.AddTextbox(1, 55, 335, 850, 60)
    With shp.TextFrame.TextRange
        .Text = Chr(128279) & " LINKS" & vbCrLf & _
                "GitHub: github.com/dv0331/Ai-Data-Analyst  |  Live Demo: [Streamlit Cloud URL]  |  Video: [Loom/YouTube Link]"
        .Font.Size = 14
        .Font.Color.RGB = RGB(255, 255, 255)
        .Paragraphs(1).Font.Bold = True
        .Paragraphs(1).Font.Size = 16
    End With
    
    ' Thank you
    Set shp = sld.Shapes.AddTextbox(1, 40, 420, 880, 50)
    With shp.TextFrame.TextRange
        .Text = "Thank you! Questions welcome during the walkthrough."
        .Font.Size = 20
        .Font.Bold = True
        .Font.Color.RGB = RGB(32, 56, 100)
        .ParagraphFormat.Alignment = 2
    End With
    
    ' Sample outputs note
    Set shp = sld.Shapes.AddTextbox(1, 650, 470, 270, 25)
    With shp.TextFrame.TextRange
        .Text = "Sample PDFs in /Samples/ folder"
        .Font.Size = 10
        .Font.Italic = True
        .Font.Color.RGB = RGB(128, 128, 128)
    End With
    
    ' ========================================
    ' Final touches
    ' ========================================
    MsgBox "Presentation created with 5 slides!" & vbCrLf & vbCrLf & _
           "Next steps:" & vbCrLf & _
           "1. Add screenshots from your app" & vbCrLf & _
           "2. Insert Architecture.svg on Slide 2" & vbCrLf & _
           "3. Customize colors to match your brand" & vbCrLf & _
           "4. Save as .pptx or .pdf", vbInformation, "Data-to-Insight Presentation"
    
End Sub

' ============================================================================
' HELPER: Run this to apply a consistent theme
' ============================================================================
Sub ApplyConsistentTheme()
    Dim sld As Object
    Dim shp As Object
    
    For Each sld In ActivePresentation.Slides
        ' Set background gradient
        sld.FollowMasterBackground = False
        sld.Background.Fill.TwoColorGradient 1, 1
        sld.Background.Fill.GradientStops(1).Color.RGB = RGB(255, 255, 255)
        sld.Background.Fill.GradientStops(2).Color.RGB = RGB(245, 248, 250)
    Next sld
    
    MsgBox "Theme applied!", vbInformation
End Sub

' ============================================================================
' HELPER: Export to PDF
' ============================================================================
Sub ExportToPDF()
    Dim filePath As String
    filePath = Application.ActivePresentation.Path & "\Data_Analyst_Presentation.pdf"
    ActivePresentation.SaveAs filePath, 32 ' ppSaveAsPDF
    MsgBox "Exported to: " & filePath, vbInformation
End Sub

