import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Lesson Designer",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Multi-Agent Lesson Design System")
st.markdown("""
Three specialist AI agents collaborate to design your lesson:
- **Agent 1 — Lesson Designer:** Creates the core structure
- **Agent 2 — Equity Reviewer:** Checks inclusion and accessibility
- **Agent 3 — Curriculum Aligner:** Maps to standards and progression

---
""")

# ─────────────────────────────────────────
# API KEY — loaded from Streamlit secrets
# if deployed, or entered by user locally
# ─────────────────────────────────────────
default_key = ""
try:
    # On Streamlit Cloud this reads from the
    # secrets manager — never exposed in code
    default_key = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    pass  # No secrets file — user will paste key below

if default_key:
    api_key = default_key
    st.success("API key loaded securely.", icon="✅")
else:
    api_key = st.text_input(
        "Your OpenRouter API key",
        type="password",
        placeholder="Paste your key here (sk-or-...)",
        help="Free key at openrouter.ai — never stored or sent anywhere except to OpenRouter"
    )

st.divider()

# ─────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────
st.subheader("Tell the agents about your lesson")

col1, col2 = st.columns(2)

with col1:
    subject = st.text_input(
        "Subject",
        placeholder="e.g. Science, English, Maths"
    )
    grade = st.text_input(
        "Grade / Year group",
        placeholder="e.g. Year 8, Grade 5"
    )

with col2:
    topic = st.text_input(
        "Topic",
        placeholder="e.g. Introduction to climate change"
    )
    duration = st.selectbox(
        "Lesson duration",
        ["30 minutes", "45 minutes", "50 minutes",
         "60 minutes", "90 minutes"]
    )

context = st.text_area(
    "Any class context? (optional)",
    placeholder="e.g. Mixed ability, 3 EAL students, "
                "1 student with dyslexia, exam class...",
    height=80
)

st.divider()

# ─────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────
if st.button("🚀 Design my lesson", use_container_width=True):

    if not subject or not grade or not topic:
        st.error("Please fill in Subject, Grade, and Topic.")
    elif not api_key:
        st.error("Please enter your OpenRouter API key.")
    else:
        # ─────────────────────────────────────────
        # LLM — FREE model via OpenRouter
        # No caching issues, no version conflicts
        # ─────────────────────────────────────────
        llm = LLM(
            model="openrouter/auto",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7
        )

        # ─────────────────────────────────────────
        # THREE AGENTS
        # ─────────────────────────────────────────
        lesson_designer = Agent(
            role="Lead Lesson Designer",
            goal="Design engaging, well-structured lessons "
                 "that achieve clear learning objectives",
            backstory="You are an experienced curriculum "
                      "designer with 20 years creating "
                      "lessons across K-12 and higher "
                      "education. You believe every lesson "
                      "needs a strong hook, clear structure, "
                      "and meaningful student activity.",
            llm=llm,
            verbose=False,
            cache=False,
            max_iter=3
        )

        equity_reviewer = Agent(
            role="Equity and Inclusion Specialist",
            goal="Ensure every lesson is accessible, "
                 "inclusive, and culturally responsive "
                 "for all learners",
            backstory="You are a SEND specialist and equity "
                      "consultant who has worked with diverse "
                      "schools globally. You review every "
                      "lesson with a critical inclusion lens "
                      "and always give concrete, practical "
                      "differentiation strategies.",
            llm=llm,
            verbose=False,
            cache=False,
            max_iter=3
        )

        curriculum_aligner = Agent(
            role="Curriculum Standards Expert",
            goal="Ensure lessons connect meaningfully to "
                 "curriculum standards and fit into the "
                 "broader learning progression",
            backstory="You are a curriculum standards "
                      "specialist who has worked with "
                      "education ministries across multiple "
                      "countries. You ensure lessons are "
                      "pedagogically sound and "
                      "assessment-valid.",
            llm=llm,
            verbose=False,
            cache=False,
            max_iter=3
        )

        # ─────────────────────────────────────────
        # TASKS
        # ─────────────────────────────────────────
        context_text = (
            f"\nClass context: {context}" if context else ""
        )

        design_task = Task(
            description=f"""Design a complete lesson plan for:
            Subject: {subject}
            Grade: {grade}
            Topic: {topic}
            Duration: {duration}
            {context_text}

            Structure your output with these clearly
            labeled sections:
            - LEARNING OBJECTIVE
            - HOOK (opening activity, 5 min)
            - MAIN INSTRUCTION (teaching strategy)
            - STUDENT ACTIVITY (hands-on task)
            - CONSOLIDATION (exit task)
            - MATERIALS NEEDED

            Be specific and practical. Name real
            resources where possible.""",
            expected_output="A complete, clearly structured "
                           "lesson plan with all six sections "
                           "labeled and specific activities.",
            agent=lesson_designer
        )

        equity_task = Task(
            description="""Review the lesson plan created
            by the Lesson Designer and add an equity and
            inclusion analysis. Provide:

            1. ACCESSIBILITY ASSESSMENT
               What works well and what needs attention
               for diverse learners

            2. CULTURAL RESPONSIVENESS CHECK
               Does the lesson reflect diverse perspectives?
               Specific suggestions to improve this.

            3. DIFFERENTIATION STRATEGIES
               - For students needing more support:
                 2 specific scaffolding strategies
               - For students needing more challenge:
                 2 specific extension activities
               - For ELL/EAL students:
                 2 specific language support strategies

            4. REVISED ACTIVITY (if needed)
               If the main activity needs changes for
               inclusion, suggest the revised version.
               If already strong, say why.""",
            expected_output="A detailed equity review with "
                           "specific, actionable strategies "
                           "for all learner types.",
            agent=equity_reviewer,
            context=[design_task]
        )

        alignment_task = Task(
            description="""Review both the lesson plan and
            equity review. Produce a curriculum alignment
            report with these sections:

            1. CURRICULUM CONNECTIONS
               Standards and goals this lesson addresses

            2. CROSS-CURRICULAR LINKS
               At least 2 connections to other subjects

            3. LEARNING PROGRESSION
               - Before this lesson: prerequisites needed
               - After this lesson: natural next topics

            4. ASSESSMENT VALIDITY CHECK
               Does the consolidation task actually measure
               the stated learning objective?

            5. QUALITY RATINGS
               Rate each out of 5 with one sentence:
               - Lesson design: X/5
               - Equity and inclusion: X/5
               - Curriculum alignment: X/5

            6. TOP 3 IMPROVEMENTS
               The three highest-impact changes the
               teacher could make.""",
            expected_output="A complete curriculum alignment "
                           "report covering all six sections "
                           "with quality ratings and three "
                           "prioritised improvements.",
            agent=curriculum_aligner,
            context=[design_task, equity_task]
        )

        # ─────────────────────────────────────────
        # ASSEMBLE AND RUN THE CREW
        # ─────────────────────────────────────────
        crew = Crew(
            agents=[lesson_designer, equity_reviewer,
                    curriculum_aligner],
            tasks=[design_task, equity_task,
                   alignment_task],
            process=Process.sequential,
            verbose=False,
            memory=False,
            cache=False
        )

        with st.status("Agents are working...",
                       expanded=True) as status:
            st.write("🎨 **Agent 1** — Lesson Designer building your lesson...")
            st.write("♿ **Agent 2** — Equity Reviewer standing by...")
            st.write("📌 **Agent 3** — Curriculum Aligner standing by...")
            st.write("⏳ This takes 30–60 seconds...")

            try:
                result = crew.kickoff()
                status.update(
                    label="✅ Lesson package ready!",
                    state="complete"
                )
            except Exception as e:
                status.update(
                    label="❌ Something went wrong",
                    state="error"
                )
                st.error(f"Error: {str(e)}")
                st.stop()

        # ─────────────────────────────────────────
        # DISPLAY RESULTS
        # ─────────────────────────────────────────
        st.divider()
        st.subheader("📋 Your Complete Lesson Package")
        st.markdown(str(result))

        st.download_button(
            label="⬇️ Download lesson as text file",
            data=str(result),
            file_name=(
                f"lesson_{subject}_{topic}.txt"
                .replace(" ", "_")
            ),
            mime="text/plain"
        )
