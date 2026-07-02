# Import the tools we need from CrewAI
from crewai import Agent, Task, Crew, Process
from crewai.llms import LLM  # Modern way in CrewAI 1.15+
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found!")

# Create the LLM
llm = LLM(
    model="groq/llama3-70b-8192",  # or "groq/llama-3.3-70b-versatile"
    api_key=GROQ_API_KEY
)

# Pass it to each agent
agent = Agent(
    role="Lead Lesson Designer",
    goal="Design engaging lesson plans",
    backstory="You are an expert educator...",
    llm=llm,  # <-- Pass LLM here
    verbose=True
)

#─────────────────────────────────────
# DEFINE THE THREE AGENTS
# Each agent has a role, a goal, and a backstory
# Think of these as their job description
# ─────────────────────────────────────────

lesson_designer = Agent(
    role="Lead Lesson Designer",
    goal="Design engaging, well-structured lessons that "
         "achieve clear learning objectives",
    backstory="You are an experienced curriculum designer "
              "with 20 years creating lessons across K-12 "
              "and higher education. You believe every "
              "lesson needs a hook, clear structure, and "
              "meaningful student activity.",
    verbose=True  # shows thinking in terminal
)

equity_reviewer = Agent(
    role="Equity and Inclusion Specialist",
    goal="Ensure every lesson is accessible, inclusive, "
         "and culturally responsive for all learners",
    backstory="You are a SEND specialist and equity "
              "consultant who has worked with diverse "
              "schools globally. You review every lesson "
              "with a critical inclusion lens and always "
              "provide concrete differentiation strategies.",
    verbose=True
)

curriculum_aligner = Agent(
    role="Curriculum Standards Expert",
    goal="Ensure lessons connect to curriculum standards "
         "and fit into the broader learning progression",
    backstory="You are a curriculum standards specialist "
              "who has worked with education ministries "
              "across multiple countries. You ensure "
              "lessons are pedagogically sound and "
              "assessment-valid.",
    verbose=True
)

# ─────────────────────────────────────────
# DEFINE THE THREE TASKS
# Each task tells an agent exactly what to do
# Notice how task 2 reads task 1's output,
# and task 3 reads both previous outputs
# ─────────────────────────────────────────

design_task = Task(
    description="""Design a complete lesson plan for:
    Subject: {subject}
    Grade: {grade}
    Topic: {topic}
    Duration: {duration}
    
    Include: learning objective, hook activity, 
    main instruction strategy, student activity, 
    consolidation task, and materials needed.""",
    expected_output="A complete structured lesson plan "
                   "with all sections clearly labeled.",
    agent=lesson_designer
)

equity_task = Task(
    description="""Review the lesson plan created by the 
    Lesson Designer and add an equity and inclusion layer.
    
    Provide:
    1. Accessibility assessment
    2. Cultural responsiveness check  
    3. Differentiation for: support needs, 
       extension, and ELL students
    4. Any revisions needed to the original activity""",
    expected_output="A detailed equity review with "
                   "specific, actionable differentiation "
                   "strategies for all learner types.",
    agent=equity_reviewer,
    context=[design_task]  # reads the designer's output
)

alignment_task = Task(
    description="""Review both the lesson plan and equity 
    review. Add curriculum alignment analysis including:
    
    1. Curriculum connections and standards met
    2. Cross-curricular links (minimum 2)
    3. Prerequisite knowledge needed
    4. Natural follow-on lessons
    5. Assessment validity check
    6. Overall quality rating out of 5 for each agent
    7. Top 3 improvements for the teacher""",
    expected_output="A complete curriculum alignment "
                   "report plus final synthesis with "
                   "quality ratings and improvements.",
    agent=curriculum_aligner,
    context=[design_task, equity_task]  # reads both
)

# ─────────────────────────────────────────
# ASSEMBLE THE CREW AND RUN IT
# Sequential process means agents run in order
# ─────────────────────────────────────────

crew = Crew(
    agents=[lesson_designer, equity_reviewer, 
            curriculum_aligner],
    tasks=[design_task, equity_task, alignment_task],
    process=Process.sequential,  # run in order
    verbose=True
)

# ─────────────────────────────────────────
# THIS IS WHERE YOU CHANGE THE LESSON INPUT
# ─────────────────────────────────────────

result = crew.kickoff(inputs={
    "subject": "Science",
    "grade": "Year 8",
    "topic": "Introduction to climate change",
    "duration": "50 minutes"
})

print("\n" + "="*50)
print("FINAL LESSON PACKAGE")
print("="*50)
print(result)